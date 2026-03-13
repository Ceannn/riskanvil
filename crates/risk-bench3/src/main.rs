use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use clap::{Parser, ValueEnum};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use crossbeam_queue::ArrayQueue;
use futures_util::stream::{FuturesUnordered, StreamExt};
use hdrhistogram::Histogram;
use http::Uri;
use memmap2::Mmap;
use mio::net::TcpStream;
use mio::{Events, Interest, Poll, Token};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::cmp::min;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, IoSlice, Read, Write};
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use std::str;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::runtime::Builder as TokioRuntimeBuilder;

include!("bench_types.rs");

include!("bench_setup.rs");

include!("bench_decode.rs");

include!("bench_h2.rs");

include!("bench_http.rs");

include!("bench_agg.rs");

include!("bench_entry.rs");
