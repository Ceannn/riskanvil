# Standalone L2 Next Battle Plan for 7945HX

## Status Update

截至 `2026-03-09`，这版新计划已经拿到两个实测结论：

1. `Tree64 / Tree128` 分 pack 已被统计判死
   - late trees 基本全在 `97-128 leaves`
   - 对当前 bundle 不值得实现
2. `one-half-live` 已实测打赢
   - 当前独立仓库先前胜者：`237430.0 rows/s` median
3. `sample-driven front-width sweep` 已继续打赢
   - `front-12` 负收益，已回滚
   - `front-16` 已继续被“均匀 active 抽样”版本打赢
   - 当前最新胜者：`265532.4 rows/s` median
   - 方法是：对全 active-row 区间均匀采样 `256` 行，贪心重排全 late trees 的前 `16` 个 block

当前独立实验仓库胜者：

- repo: `/home/ceann/projects/rust/quickscorer_7945hx_minpack_20260308/rust_quickl1`
- commit: `44a6cab`
- tag: `win-20260310-ac-265532-uniform-front16-reorder`

## 0. 结论先写死

上一版 `Prefix Program v4` 的主假设已经基本被实测证伪。

已经证伪的方向包括：

1. `late span program`
2. `callback -> certify` 结构改写
3. `cert LUT` 大展开
4. `software prefetch`
5. `hot fat / cold thin` 的最小 `subpack` 复制版

所以接下来的路线不再继续围绕：

1. callback / span machinery
2. certify LUT 化
3. full / semi-full pack 展开

而是换到更接近内核本体的七刀。

一句话：

**不要再把主要希望放在“解释器外层重写”上，而要去吃 tree/block/leaf 这三层天然的交换律窗口和瘦身空间。**

---

## 1. 第一刀：吃满两层“交换律窗口”

当前有两层天然自由度：

### 1.1 tree 内 block 顺序可重排

本质上 `late prefix` 做的是 mask 交集。

叶子只在树末尾被 decode，所以 tree 内 block 顺序大概率可离线重排，只要保持逻辑候选集合语义不变。

### 1.2 checkpoint 段内 tree 计算顺序可与提交顺序分离

不能直接乱加分数，因为 `f32` 加法有顺序效应。

但可以：

```rust
for t in seg.eval_order {
    seg_contrib[t.orig_pos] = eval_tree_specialized(t, ...);
}
for pos in 0..seg_len {
    score += seg_contrib[pos];
}
certify(score);
```

也就是说：

1. 按机器喜欢的顺序算
2. 按原语义顺序记账

这比继续抠 closure 或 raw pointer 更像正解。

---

## 2. 第二刀：按 leaf width 分 pack

如果 `late prefix` 里相当一部分树只有 `<=64` 个叶子，那么统一用 `lo/hi` 两个 `u64` 的 `128-bit` 形态，就是明显冗余。

建议至少分成两类：

1. `Tree64`
   - `u64 candidate`
   - `Vec<u64> masks`

2. `Tree128`
   - `lo/hi`
   - `Vec<Mask128>`

这不是 `full expand`，而是很干净的瘦身。

它同时砍三件事：

1. mask 带宽
2. `AND` 指令数
3. `resolved` 判定复杂度

### 2.1 先做 histogram

先统计：

`late trees by leaf_count_bucket`

如果 `<=64` 占大头，这刀可能是当前最大的金矿。

---

## 3. 第三刀：128-bit 树加 one-half-live fast path

这刀很值得认真打。

因为一旦出现：

1. `lo == 0`
2. 或 `hi == 0`

另一半永远不会复活。

后续实际上已经掉进 `64-bit` 世界，只是代码还在继续按 `128-bit` 世界跑。

### 3.1 建议做法

把 mask 拆成：

1. `masks_lo: Vec<u64>`
2. `masks_hi: Vec<u64>`

运行时状态分成：

1. `Both`
2. `LoOnly`
3. `HiOnly`

一旦 collapse：

1. 后续只读一边 mask
2. 后续只做一个 `&=`
3. 后续 `resolved` 只看一个 `u64`

真正让 `resolved` 变便宜的，不是把 `is_power_of_two()` 换个写法，而是尽快从 `128-bit` 掉到 `64-bit`。

这刀和上一刀是连招。

---

## 4. 第四刀：tree 内 block 顺序按“尽早单叶/半塌缩”重排

当前真正贵的不是某个 block 多两条指令，而是一棵树平均要吃多少个 block 才 resolve。

所以 tree 内 block 顺序不该按结构直觉排，而该离线按样本集做 greedy 重排。

目标函数不是“单 block 周期数”，而是：

1. `P(singleton after k)`
2. `E[survivor_bits after block]`
3. `P(lo_zero xor hi_zero)`
4. 或直接最小化 `E[blocks_until_resolve]`

这刀的味道不是给发动机抛光，而是直接减少活塞往复次数。

而且它不碰 calibration / checkpoint 语义，是非常少见的高 ROI 区。

---

## 5. 第五刀：单行低延迟场景，先做 2~4 棵树交错执行

不要先赌 `AVX-512`。

当前 loop 最大问题之一，是单树内依赖链太像 pointer chasing。

更直接的破局办法不是把单条链变宽，而是让核心同时盯几条独立链。

### 5.1 checkpoint 段内 2~4 trees in flight

每棵树维护自己的：

1. `lo/hi`
2. `block_ptr`
3. `state`

然后 round-robin 每次推进一个 block。

树完成后把 leaf contribution 填到：

`seg_contrib[orig_pos]`

段末再按原顺序提交：

```rust
for pos in 0..seg_len {
    score += seg_contrib[pos];
}
```

### 5.2 这刀的意义

它不是为了 SIMD，而是为了给 `OoO` 核制造：

1. `MLP`
2. `ILP`

现在像一根吸管喂数据。

交错后像四根吸管并排灌。

而且你已经有 checkpoint segment 这个天然边界，非常适合做 `segment-local scheduler`。

---

## 6. 第六刀：Pack 重排走“瘦身型”，别再搞 full expand

`full rank -> mask` 展开已经被证伪，这不意外。

这类 bitvector scorer 很容易被工作集体积反杀。

所以 pack 重排原则是“瘦身”，不是“堆更多预计算”。

### 6.1 正确的层级

按这个层级重排：

1. `checkpoint segment`
2. `tree kind`
3. `hot fields`

### 6.2 具体建议

1. 以 checkpoint segment 为基本包，而不是整个 late pack 一锅端
2. segment 内按 `Tree64 / Tree128` 分开
3. `block_hdr` 做 hot/cold split，只留热字段在主循环 cache line
4. `fid` 尽量压到 `u16`
5. `mask_off / lut_off` 尽量变成 tree-local 或 seg-local 小偏移
6. `masks_lo / masks_hi` 分离
7. `leaf_values` 按 tree kind 分池
8. 如果 bucket cardinality histogram 很小，再考虑 nibble-packed LUT

原则不是“字段挪一挪”，而是：

**让主循环只碰到它非碰不可的热字节。**

---

## 7. 第七刀：真要大改，就往 feature-major checkpoint slab 走

这是另一条世界线。

原版 `QS` 论文强调的是按 feature 线性扫描，减少依赖并利于 prefetch。

`BWQS` 也强调把数据切成适合 cache 的块。

当前 `late prefix` 是 tree-major，而且每个 block 都在走：

`rank -> LUT -> bucket -> mask`

这其实有点逆着 `QS` 最初的几何形状跑。

### 7.1 如果接受结构性改写

建议认真考虑：

1. 以 checkpoint segment 为 slab
2. slab 内维护该段 trees 的 candidate bitsets
3. 按 feature 扫描该段 block lists
4. 段末统一 decode leaf
5. 再按原顺序提交 score

这会多占一点临时 bitset 状态，但有机会把当前最恶心的 per-tree 依赖链改造成更线性的访问形态。

这个方向更像 `BWQS / QS` 正统续命，而不是继续在单树小循环里拧螺丝。

---

## 8. 直接回答当前 2 个核心问题

### 8.1 两段间接访存怎么优化

最靠谱的不是继续 raw-pointer 化，而是：

1. tree 内 block 重排，减少平均 block 数
2. `Tree64 / Tree128` 分 pack
3. `one-half-live`，尽快掉进 `64-bit`
4. checkpoint 段内多树交错
5. 真大改就 `feature-major slab`

### 8.2 不显著扩大工作集的前提下，QsPack 怎么重排

按这个层级：

1. `checkpoint segment`
2. `tree kind`
3. `hot fields`

核心不是 `SoA / AoS` 宗教战，而是：

**让主循环访问的东西尽量变成局部小池。**

`BWQS` 的核心思路本质也是这个：块要小到 cache 喜欢。

---

## 9. 推荐实际执行顺序

### Phase A：先做统计和剖面

先补离线统计，不改内核行为：

1. `late trees by leaf_count_bucket`
2. `block count / tree`
3. `blocks_until_resolve` 分布
4. `lo==0 xor hi==0` 出现率
5. checkpoint segment 内 tree/block 热度

如果 `Tree64` 占比不高，就不要上第二刀和第三刀。

### Phase B：`Tree64 / Tree128` 分 pack

先做最小结构刀：

1. `Tree64` 单独内核
2. `Tree128` 保持现有内核
3. 不碰 checkpoint 语义

这是当前最干净的一刀。

### Phase C：`one-half-live`

在 `Tree128` 上加：

1. `Both`
2. `LoOnly`
3. `HiOnly`

尽快从 `128-bit` 降到 `64-bit`。

### Phase D：tree 内 block 重排

目标：

1. 提前单叶
2. 提前半塌缩
3. 降低 `E[blocks_until_resolve]`

### Phase E：segment 内多树交错

先从 `2 trees in flight` 开始，不要直接上更复杂版本。

### Phase F：如果前面都没有明显收益，再讨论 feature-major slab

这一步是新世界线，不应在当前小修路线失败前贸然开工。

---

## 10. 最终判断

接下来最不该继续做的是：

1. 再写 callback/span/cert machinery 改写
2. 再做大体积 cert LUT
3. 再做 full expand
4. 再赌手工 prefetch

最该做的是：

1. 吃满交换律窗口
2. 先做 `Tree64 / Tree128` 瘦身
3. 再做 `one-half-live`
4. 再做 block reorder
5. 再做 segment-local 多树交错

这条路线比上一版更接近 `late prefix` 内核本体，也更符合当前已经被实测证伪后的事实边界。

---

## 11. 2026-03-10 实测结论更新：放弃 QS late kernel 是真钱

当前这条作战线后续又补了两类实验：

1. `feature-major checkpoint slab`
2. `late prefix if-tree specialization`

结论非常明确：

- `feature-major checkpoint slab` 已被实测证伪。
  - `120k x 20` median 只有 `141,979.7 rows/s`
  - 行为指标没漂，但执行形态太重，整体大幅退化
- `late prefix if-tree specialization` 打穿了当前胜者。
  - 保留现有 hot prefix、atlas certify、exact continuation 不动
  - 只把 `nan_free` 的 late prefix 从 QS kernel 换成专门化 if-tree program
  - `120k x 20` median 提到 `600,590.7 rows/s`
  - 相比此前胜者 `265,532.4 rows/s`，提升 `+126.19%`

这说明一件事：

**当前 standalone L2 最大的结构拖累，确实就是 QS late kernel 本身。**

不是 certify 本体，不是 hot prefix，也不是 tail。
在 `7945HX` 上，late prefix 的传统树遍历程序形态明显比：

`rank -> LUT -> bucket -> mask -> lo/hi &= mask`

更适合这条单行 fast path。

### 当前建议

后续性能主线应当转成：

1. 保留 hot prefix 的 compiled exact 路径
2. 保留现有 certify 逻辑
3. 保留 exact continuation 作为 tail
4. **正式把 late prefix 的主攻方向切到 if-tree specialization**

如果继续 refine，这条线后面值得做的是：

1. 继续专门化 late if-tree program 的节点布局和 tree-local layout
2. 评估是否把 late if-tree 也编译成更紧的 `compiled node` 形式
3. 只在这条新 late kernel 上继续做 7945HX 定向优化

### 2026-03-10 追加实测

上面第 1 条已经拿到了正收益：

- 观察到 late segment:
  - `1856` 棵树
  - 总节点 `472638`
  - **单棵树最大只有 `255` 个节点**
  - late 覆盖特征只有 `86`
- 因此把 late if-tree program 进一步压成了 tree-local compact node：
  - `fidx -> u8`
  - `left/right -> u8`
  - 按树内本地索引走 child
- `120k x 20` median 从 `600,590.7 rows/s` 再提到 `621,510.8 rows/s`

这说明这条 if-tree 世界线后续仍然有可挖空间，而且方向是：

**继续缩窄 late 节点布局和 tree-local 执行状态，而不是回头救 QS late kernel。**
