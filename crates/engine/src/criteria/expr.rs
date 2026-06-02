//! Typed expression language for signal-function criteria (S3).
//!
//! A criterion can filter matches (`where`) and place anchors (`point_expr` /
//! `span_expr`) with small **expressions** over signals — e.g.
//! `mean(f0 over interval) > 1.2 * mean(f0, file)` or `argmax(intensity)`.
//! This is the generalization the user asked for: rather than a fixed menu of
//! signal functions, one typed expression language over two open registries
//! (signals + reducers) covers filters, anchors, and span endpoints alike.
//!
//! Design: the 2026-05-31 DEVLOG entries. The evaluator is **pure** — the
//! caller ([`crate::corpus::Project`]) pre-computes the referenced signal
//! series into a [`SignalSet`] and hands it in, exactly as the S2 interval
//! evaluator takes pre-fetched intervals.
//!
//! Value model: every expression evaluates to a [`Value`] — `Num` (a scalar,
//! or a time in seconds) or `Bool`. A role interprets the result: `where`
//! wants `Bool`; an anchor wants `Num` (read as seconds). A reduction over an
//! empty region of interest (e.g. f0 across a fully-unvoiced interval) is
//! **undefined** — it evaluates to `None` and propagates, and the per-match
//! caller drops that match (no proposal). A type error (a `Bool` where a
//! number is needed, an unknown signal/function) is a hard error.
//!
//! Grammar (recursive descent; hand-rolled, no dependencies):
//! ```text
//! or   → and ("or" and)*
//! and  → cmp ("and" cmp)*
//! cmp  → sum (("<"|"<="|">"|">="|"=="|"!=") sum)?
//! sum  → term (("+"|"-") term)*
//! term → factor (("*"|"/") factor)*
//! factor → number ("ms"|"%")? | call | keyword | "(" or ")" | ("-"|"not") factor
//! call → ident "(" ident ("," expr)* ("," ident)* ")"
//! ```
//! Number suffixes: `ms` (→ seconds), `%` (→ that fraction of the matched
//! interval's `duration`). Keywords: `start`, `end`, `duration` (seconds),
//! `true`, `false`.

use std::collections::HashMap;

/// A time-sampled signal: parallel `times` (seconds) and `values` (the
/// signal's native unit — Hz for `f0`, dB for `intensity`, …). Samples are
/// assumed sorted by time.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SampledSignal {
    /// Sample times in seconds, ascending.
    pub times: Vec<f64>,
    /// Sample values, parallel to `times`.
    pub values: Vec<f64>,
}

/// The signals available to an expression, keyed by the name used in the
/// expression text (e.g. `"f0"`, `"intensity"`, or a measure-track tier name).
pub type SignalSet = HashMap<String, SampledSignal>;

/// A runtime value: a scalar/time (`Num`) or a boolean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    /// A number — a bare scalar, or a time in seconds for anchor roles.
    Num(f64),
    /// A boolean — the result of a comparison or logical op.
    Bool(bool),
}

/// Reductions over a signal's samples within a scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReduceFunc {
    Mean,
    Max,
    Min,
    Median,
    Std,
    Range,
    ArgMax,
    ArgMin,
    FirstCrossing,
    LastCrossing,
}

impl ReduceFunc {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "mean" => Self::Mean,
            "max" => Self::Max,
            "min" => Self::Min,
            "median" => Self::Median,
            "std" => Self::Std,
            "range" => Self::Range,
            "argmax" => Self::ArgMax,
            "argmin" => Self::ArgMin,
            "first_crossing" => Self::FirstCrossing,
            "last_crossing" => Self::LastCrossing,
            _ => return None,
        })
    }
    /// Whether this reducer takes a threshold argument (the crossings do).
    fn takes_threshold(self) -> bool {
        matches!(self, Self::FirstCrossing | Self::LastCrossing)
    }
}

/// The region of a signal a reducer ranges over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Scope {
    /// The matched interval `[start, end]` (the default).
    #[default]
    Interval,
    /// The whole signal (the file).
    File,
}

/// Crossing direction for `first_crossing` / `last_crossing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    Rising,
    Falling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Keyword {
    Start,
    End,
    Duration,
}

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Num(f64),
    /// `n%` — `n/100` of the matched interval's duration (resolved at eval).
    Percent(f64),
    Bool(bool),
    Keyword(Keyword),
    Neg(Box<Node>),
    Not(Box<Node>),
    Bin(BinOp, Box<Node>, Box<Node>),
    Reduce {
        func: ReduceFunc,
        signal: String,
        threshold: Option<Box<Node>>,
        scope: Scope,
        direction: Option<Dir>,
    },
}

/// A parsed, reusable expression. Parse once with [`Expr::parse`], then
/// [`Expr::eval`] per match.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    root: Node,
}

/// Per-match evaluation context: the matched interval's bounds (seconds) and
/// the available signals.
pub struct EvalCtx<'a> {
    /// Matched interval start, seconds.
    pub start: f64,
    /// Matched interval end, seconds.
    pub end: f64,
    /// Pre-computed signals, keyed by name.
    pub signals: &'a SignalSet,
}

impl Expr {
    /// Parses an expression from text. Errors describe the first problem.
    pub fn parse(src: &str) -> Result<Self, String> {
        let tokens = tokenize(src)?;
        let mut p = Parser { tokens, pos: 0 };
        let root = p.parse_or()?;
        if p.pos != p.tokens.len() {
            return Err(format!("unexpected trailing input near token {}", p.pos));
        }
        Ok(Expr { root })
    }

    /// The distinct signal names this expression references (the first argument
    /// of every reducer call), in first-seen order. The caller computes these.
    pub fn signals(&self) -> Vec<String> {
        let mut out = Vec::new();
        collect_signals(&self.root, &mut out);
        out
    }

    /// Evaluates against `ctx`. `Ok(Some(v))` is a defined result; `Ok(None)`
    /// means the expression is *undefined* over this match (an empty
    /// reduction) and the match should be skipped; `Err` is an authoring/type
    /// error that should abort the run.
    pub fn eval(&self, ctx: &EvalCtx) -> Result<Option<Value>, String> {
        eval_node(&self.root, ctx)
    }
}

fn collect_signals(node: &Node, out: &mut Vec<String>) {
    match node {
        Node::Reduce {
            signal, threshold, ..
        } => {
            if !out.contains(signal) {
                out.push(signal.clone());
            }
            if let Some(t) = threshold {
                collect_signals(t, out);
            }
        }
        Node::Neg(x) | Node::Not(x) => collect_signals(x, out),
        Node::Bin(_, a, b) => {
            collect_signals(a, out);
            collect_signals(b, out);
        }
        Node::Num(_) | Node::Percent(_) | Node::Bool(_) | Node::Keyword(_) => {}
    }
}

// ====================================================================
// Evaluation
// ====================================================================

fn as_num(v: Value) -> Result<f64, String> {
    match v {
        Value::Num(n) => Ok(n),
        Value::Bool(_) => Err("expected a number but got a boolean".into()),
    }
}

fn as_bool(v: Value) -> Result<bool, String> {
    match v {
        Value::Bool(b) => Ok(b),
        Value::Num(_) => Err("expected a boolean but got a number".into()),
    }
}

/// Evaluates two operands, returning `None` if either is undefined.
fn eval_pair(a: &Node, b: &Node, ctx: &EvalCtx) -> Result<Option<(Value, Value)>, String> {
    let (Some(x), Some(y)) = (eval_node(a, ctx)?, eval_node(b, ctx)?) else {
        return Ok(None);
    };
    Ok(Some((x, y)))
}

fn eval_node(node: &Node, ctx: &EvalCtx) -> Result<Option<Value>, String> {
    Ok(match node {
        Node::Num(n) => Some(Value::Num(*n)),
        Node::Percent(frac) => Some(Value::Num(frac * (ctx.end - ctx.start))),
        Node::Bool(b) => Some(Value::Bool(*b)),
        Node::Keyword(k) => Some(Value::Num(match k {
            Keyword::Start => ctx.start,
            Keyword::End => ctx.end,
            Keyword::Duration => ctx.end - ctx.start,
        })),
        Node::Neg(x) => match eval_node(x, ctx)? {
            Some(v) => Some(Value::Num(-as_num(v)?)),
            None => None,
        },
        Node::Not(x) => match eval_node(x, ctx)? {
            Some(v) => Some(Value::Bool(!as_bool(v)?)),
            None => None,
        },
        Node::Bin(op, a, b) => {
            let Some((x, y)) = eval_pair(a, b, ctx)? else {
                return Ok(None);
            };
            eval_bin(*op, x, y)?
        }
        Node::Reduce {
            func,
            signal,
            threshold,
            scope,
            direction,
        } => eval_reduce(*func, signal, threshold.as_deref(), *scope, *direction, ctx)?,
    })
}

fn eval_bin(op: BinOp, x: Value, y: Value) -> Result<Option<Value>, String> {
    Ok(match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
            let (a, b) = (as_num(x)?, as_num(y)?);
            let r = match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                // Division by zero is undefined (skip), not an error.
                BinOp::Div => {
                    if b == 0.0 {
                        return Ok(None);
                    }
                    a / b
                }
                _ => unreachable!(),
            };
            Some(Value::Num(r))
        }
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
            let (a, b) = (as_num(x)?, as_num(y)?);
            Some(Value::Bool(match op {
                BinOp::Lt => a < b,
                BinOp::Le => a <= b,
                BinOp::Gt => a > b,
                BinOp::Ge => a >= b,
                _ => unreachable!(),
            }))
        }
        BinOp::Eq | BinOp::Ne => {
            // Equality works for two numbers or two booleans.
            let eq = match (x, y) {
                (Value::Num(a), Value::Num(b)) => a == b,
                (Value::Bool(a), Value::Bool(b)) => a == b,
                _ => return Err("cannot compare a number with a boolean".into()),
            };
            Some(Value::Bool(if op == BinOp::Eq { eq } else { !eq }))
        }
        BinOp::And => Some(Value::Bool(as_bool(x)? && as_bool(y)?)),
        BinOp::Or => Some(Value::Bool(as_bool(x)? || as_bool(y)?)),
    })
}

/// Returns the indices `[lo, hi)` of `times` within `[start, end]` (inclusive),
/// or the whole range for `Scope::File`.
fn scoped_slice<'a>(sig: &'a SampledSignal, scope: Scope, ctx: &EvalCtx) -> &'a [f64] {
    match scope {
        Scope::File => &sig.values,
        Scope::Interval => {
            let lo = sig.times.partition_point(|&t| t < ctx.start);
            let hi = sig.times.partition_point(|&t| t <= ctx.end);
            &sig.values[lo..hi]
        }
    }
}

/// The `(times, values)` slice for a scope (crossings need both).
fn scoped_pair<'a>(sig: &'a SampledSignal, scope: Scope, ctx: &EvalCtx) -> (&'a [f64], &'a [f64]) {
    match scope {
        Scope::File => (&sig.times, &sig.values),
        Scope::Interval => {
            let lo = sig.times.partition_point(|&t| t < ctx.start);
            let hi = sig.times.partition_point(|&t| t <= ctx.end);
            (&sig.times[lo..hi], &sig.values[lo..hi])
        }
    }
}

fn eval_reduce(
    func: ReduceFunc,
    signal: &str,
    threshold: Option<&Node>,
    scope: Scope,
    direction: Option<Dir>,
    ctx: &EvalCtx,
) -> Result<Option<Value>, String> {
    let sig = ctx
        .signals
        .get(signal)
        .ok_or_else(|| format!("unknown signal {signal:?}"))?;

    if func.takes_threshold() {
        let thr_node = threshold.ok_or_else(|| "crossing requires a threshold".to_string())?;
        let thr = match eval_node(thr_node, ctx)? {
            Some(v) => as_num(v)?,
            None => return Ok(None),
        };
        let (times, values) = scoped_pair(sig, scope, ctx);
        return Ok(find_crossing(times, values, thr, func, direction).map(Value::Num));
    }

    // Scalar / argmax / argmin reducers.
    match func {
        ReduceFunc::ArgMax | ReduceFunc::ArgMin => {
            let (times, values) = scoped_pair(sig, scope, ctx);
            if values.is_empty() {
                return Ok(None);
            }
            let mut best = 0usize;
            for i in 1..values.len() {
                let better = if func == ReduceFunc::ArgMax {
                    values[i] > values[best]
                } else {
                    values[i] < values[best]
                };
                if better {
                    best = i;
                }
            }
            Ok(Some(Value::Num(times[best])))
        }
        _ => {
            let vals = scoped_slice(sig, scope, ctx);
            if vals.is_empty() {
                return Ok(None);
            }
            let n = vals.len() as f64;
            let r = match func {
                ReduceFunc::Mean => vals.iter().sum::<f64>() / n,
                ReduceFunc::Max => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ReduceFunc::Min => vals.iter().cloned().fold(f64::INFINITY, f64::min),
                ReduceFunc::Range => {
                    let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mn = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    mx - mn
                }
                ReduceFunc::Median => median(vals),
                ReduceFunc::Std => {
                    // Population standard deviation.
                    let mean = vals.iter().sum::<f64>() / n;
                    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                    var.sqrt()
                }
                _ => unreachable!("crossing/argmax handled above"),
            };
            Ok(Some(Value::Num(r)))
        }
    }
}

fn median(vals: &[f64]) -> f64 {
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Finds a threshold crossing time by linear interpolation between the
/// bracketing samples. `first_crossing` scans forward, `last_crossing` back.
/// `direction` constrains to rising / falling crossings; `None` accepts either.
fn find_crossing(
    times: &[f64],
    values: &[f64],
    thr: f64,
    func: ReduceFunc,
    direction: Option<Dir>,
) -> Option<f64> {
    let crosses = |a: f64, b: f64| -> Option<Dir> {
        // a is below/at and b is above → rising; a above/at and b below → falling.
        if a < thr && b >= thr {
            Some(Dir::Rising)
        } else if a > thr && b <= thr {
            Some(Dir::Falling)
        } else {
            None
        }
    };
    let interp = |i: usize| -> f64 {
        let (a, b) = (values[i], values[i + 1]);
        let (ta, tb) = (times[i], times[i + 1]);
        if (b - a).abs() < f64::EPSILON {
            ta
        } else {
            ta + (thr - a) / (b - a) * (tb - ta)
        }
    };
    let n = values.len();
    if n < 2 {
        return None;
    }
    let indices: Vec<usize> = if func == ReduceFunc::LastCrossing {
        (0..n - 1).rev().collect()
    } else {
        (0..n - 1).collect()
    };
    for i in indices {
        if let Some(dir) = crosses(values[i], values[i + 1]) {
            if direction.is_none() || direction == Some(dir) {
                return Some(interp(i));
            }
        }
    }
    None
}

// ====================================================================
// Tokenizer
// ====================================================================

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Percent,
    LParen,
    RParen,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                out.push(Tok::Star);
                i += 1;
            }
            '/' => {
                out.push(Tok::Slash);
                i += 1;
            }
            '%' => {
                out.push(Tok::Percent);
                i += 1;
            }
            '<' | '>' | '=' | '!' => {
                let next_eq = chars.get(i + 1) == Some(&'=');
                let tok = match (c, next_eq) {
                    ('<', true) => Tok::Le,
                    ('<', false) => Tok::Lt,
                    ('>', true) => Tok::Ge,
                    ('>', false) => Tok::Gt,
                    ('=', true) => Tok::EqEq,
                    ('!', true) => Tok::Ne,
                    ('=', false) => return Err("'=' must be '==' for equality".into()),
                    ('!', false) => return Err("'!' must be '!=' (use 'not' for negation)".into()),
                    _ => unreachable!(),
                };
                out.push(tok);
                i += if next_eq { 2 } else { 1 };
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                let n: f64 = text
                    .parse()
                    .map_err(|_| format!("invalid number {text:?}"))?;
                out.push(Tok::Num(n));
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                out.push(Tok::Ident(chars[start..i].iter().collect()));
            }
            other => return Err(format!("unexpected character {other:?}")),
        }
    }
    Ok(out)
}

// ====================================================================
// Parser (recursive descent)
// ====================================================================

struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> Result<(), String> {
        if self.peek() == Some(t) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {t:?}, found {:?}", self.peek()))
        }
    }

    /// Consumes a keyword identifier exactly equal to `kw`.
    fn eat_ident_kw(&mut self, kw: &str) -> bool {
        if let Some(Tok::Ident(s)) = self.peek() {
            if s == kw {
                self.pos += 1;
                return true;
            }
        }
        false
    }

    fn parse_or(&mut self) -> Result<Node, String> {
        let mut lhs = self.parse_and()?;
        while self.eat_ident_kw("or") {
            let rhs = self.parse_and()?;
            lhs = Node::Bin(BinOp::Or, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Node, String> {
        let mut lhs = self.parse_cmp()?;
        while self.eat_ident_kw("and") {
            let rhs = self.parse_cmp()?;
            lhs = Node::Bin(BinOp::And, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> Result<Node, String> {
        let lhs = self.parse_sum()?;
        let op = match self.peek() {
            Some(Tok::Lt) => BinOp::Lt,
            Some(Tok::Le) => BinOp::Le,
            Some(Tok::Gt) => BinOp::Gt,
            Some(Tok::Ge) => BinOp::Ge,
            Some(Tok::EqEq) => BinOp::Eq,
            Some(Tok::Ne) => BinOp::Ne,
            _ => return Ok(lhs),
        };
        self.pos += 1;
        let rhs = self.parse_sum()?;
        Ok(Node::Bin(op, Box::new(lhs), Box::new(rhs)))
    }

    fn parse_sum(&mut self) -> Result<Node, String> {
        let mut lhs = self.parse_term()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => BinOp::Add,
                Some(Tok::Minus) => BinOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_term()?;
            lhs = Node::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<Node, String> {
        let mut lhs = self.parse_factor()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => BinOp::Mul,
                Some(Tok::Slash) => BinOp::Div,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_factor()?;
            lhs = Node::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_factor(&mut self) -> Result<Node, String> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.pos += 1;
                Ok(Node::Neg(Box::new(self.parse_factor()?)))
            }
            Some(Tok::Ident(s)) if s == "not" => {
                self.pos += 1;
                Ok(Node::Not(Box::new(self.parse_factor()?)))
            }
            Some(Tok::LParen) => {
                self.pos += 1;
                let inner = self.parse_or()?;
                self.eat(&Tok::RParen)?;
                Ok(inner)
            }
            Some(Tok::Num(n)) => {
                let n = *n;
                self.pos += 1;
                // Unit suffix: ms → seconds, % → fraction of duration.
                if self.peek() == Some(&Tok::Percent) {
                    self.pos += 1;
                    Ok(Node::Percent(n / 100.0))
                } else if self.eat_ident_kw("ms") {
                    Ok(Node::Num(n / 1000.0))
                } else {
                    Ok(Node::Num(n))
                }
            }
            Some(Tok::Ident(_)) => self.parse_ident_atom(),
            other => Err(format!("unexpected token {other:?}")),
        }
    }

    /// An identifier atom: a keyword (`start`/`end`/`duration`/`true`/`false`)
    /// or a reducer call (`func(signal, …)`).
    fn parse_ident_atom(&mut self) -> Result<Node, String> {
        let Some(Tok::Ident(name)) = self.bump() else {
            unreachable!("caller checked for an ident");
        };
        // A function call?
        if self.peek() == Some(&Tok::LParen) {
            return self.parse_call(&name);
        }
        match name.as_str() {
            "start" => Ok(Node::Keyword(Keyword::Start)),
            "end" => Ok(Node::Keyword(Keyword::End)),
            "duration" => Ok(Node::Keyword(Keyword::Duration)),
            "true" => Ok(Node::Bool(true)),
            "false" => Ok(Node::Bool(false)),
            other => Err(format!(
                "unknown name {other:?} (expected start/end/duration, a literal, \
                 or a reducer call like mean(signal))"
            )),
        }
    }

    fn parse_call(&mut self, name: &str) -> Result<Node, String> {
        let func =
            ReduceFunc::from_name(name).ok_or_else(|| format!("unknown function {name:?}"))?;
        self.eat(&Tok::LParen)?;
        // First arg: the signal name (a bare identifier).
        let signal = match self.bump() {
            Some(Tok::Ident(s)) => s,
            other => return Err(format!("{name}: expected a signal name, found {other:?}")),
        };
        let mut threshold = None;
        if func.takes_threshold() {
            self.eat(&Tok::Comma)
                .map_err(|_| format!("{name} requires a threshold argument"))?;
            threshold = Some(Box::new(self.parse_or()?));
        }
        // Optional trailing bare identifiers: a scope and/or a direction.
        let mut scope = Scope::Interval;
        let mut direction = None;
        while self.peek() == Some(&Tok::Comma) {
            self.pos += 1;
            let kw = match self.bump() {
                Some(Tok::Ident(s)) => s,
                other => return Err(format!("{name}: expected scope/direction, found {other:?}")),
            };
            match kw.as_str() {
                "interval" => scope = Scope::Interval,
                "file" => scope = Scope::File,
                "rising" if func.takes_threshold() => direction = Some(Dir::Rising),
                "falling" if func.takes_threshold() => direction = Some(Dir::Falling),
                other => {
                    return Err(format!(
                        "{name}: unexpected argument {other:?} \
                         (expected interval/file{})",
                        if func.takes_threshold() {
                            " or rising/falling"
                        } else {
                            ""
                        }
                    ));
                }
            }
        }
        self.eat(&Tok::RParen)?;
        Ok(Node::Reduce {
            func,
            signal,
            threshold,
            scope,
            direction,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(times: &[f64], values: &[f64]) -> SampledSignal {
        SampledSignal {
            times: times.to_vec(),
            values: values.to_vec(),
        }
    }

    fn ctx_with<'a>(start: f64, end: f64, signals: &'a SignalSet) -> EvalCtx<'a> {
        EvalCtx {
            start,
            end,
            signals,
        }
    }

    #[test]
    fn arithmetic_and_keywords_and_units() {
        let signals = SignalSet::new();
        let ctx = ctx_with(1.0, 3.0, &signals); // duration 2.0
        let cases = [
            ("1 + 2 * 3", 7.0),
            ("(1 + 2) * 3", 9.0),
            ("start + duration", 3.0),
            ("end - start", 2.0),
            ("start + 20ms", 1.02),
            ("start + 50%", 2.0), // 0.5 * 2.0 = 1.0; +start
            ("-duration", -2.0),
        ];
        for (src, want) in cases {
            let got = Expr::parse(src).unwrap().eval(&ctx).unwrap().unwrap();
            assert_eq!(got, Value::Num(want), "{src}");
        }
    }

    #[test]
    fn comparisons_and_logic() {
        let signals = SignalSet::new();
        let ctx = ctx_with(0.0, 1.0, &signals);
        let t = |src: &str| match Expr::parse(src).unwrap().eval(&ctx).unwrap().unwrap() {
            Value::Bool(b) => b,
            _ => panic!("expected bool from {src}"),
        };
        assert!(t("1 < 2"));
        assert!(!t("2 < 1"));
        assert!(t("1 < 2 and 3 >= 3"));
        assert!(t("1 > 2 or not (5 == 6)"));
        assert!(!t("true and false"));
    }

    #[test]
    fn reducers_over_interval_and_file_scope() {
        let mut signals = SignalSet::new();
        // times 0..4 at 1s spacing; values 10,20,30,40.
        signals.insert(
            "x".into(),
            sig(&[0.0, 1.0, 2.0, 3.0], &[10.0, 20.0, 30.0, 40.0]),
        );
        // Interval [1,2] selects samples at t=1,2 → values 20,30.
        let ctx = ctx_with(1.0, 2.0, &signals);
        let n = |src: &str| match Expr::parse(src).unwrap().eval(&ctx).unwrap().unwrap() {
            Value::Num(v) => v,
            _ => panic!("expected num from {src}"),
        };
        assert_eq!(n("mean(x)"), 25.0);
        assert_eq!(n("max(x)"), 30.0);
        assert_eq!(n("min(x)"), 20.0);
        assert_eq!(n("range(x)"), 10.0);
        assert_eq!(n("argmax(x)"), 2.0); // time of value 30
        assert_eq!(n("argmin(x)"), 1.0);
        // File scope ignores the interval bounds.
        assert_eq!(n("mean(x, file)"), 25.0); // (10+20+30+40)/4
        assert_eq!(n("max(x, file)"), 40.0);
    }

    #[test]
    fn empty_reduction_is_undefined_and_propagates() {
        let mut signals = SignalSet::new();
        signals.insert("x".into(), sig(&[0.0, 5.0], &[1.0, 2.0]));
        // Interval [2,3] contains no samples.
        let ctx = ctx_with(2.0, 3.0, &signals);
        assert_eq!(Expr::parse("mean(x)").unwrap().eval(&ctx).unwrap(), None);
        // Undefined propagates through arithmetic and comparison.
        assert_eq!(
            Expr::parse("mean(x) + 1").unwrap().eval(&ctx).unwrap(),
            None
        );
        assert_eq!(
            Expr::parse("mean(x) > 0").unwrap().eval(&ctx).unwrap(),
            None
        );
    }

    #[test]
    fn crossings_with_direction_and_interpolation() {
        let mut signals = SignalSet::new();
        // rises through 15 between t=0 (10) and t=1 (20); falls through 15
        // between t=2 (20) and t=3 (10).
        signals.insert(
            "x".into(),
            sig(&[0.0, 1.0, 2.0, 3.0], &[10.0, 20.0, 20.0, 10.0]),
        );
        let ctx = ctx_with(0.0, 3.0, &signals);
        let n = |src: &str| match Expr::parse(src).unwrap().eval(&ctx).unwrap() {
            Some(Value::Num(v)) => Some(v),
            None => None,
            _ => panic!("expected num/none from {src}"),
        };
        // First crossing of 15 (rising) interpolates to t=0.5.
        assert_eq!(n("first_crossing(x, 15, rising)"), Some(0.5));
        // Falling crossing interpolates to t=2.5.
        assert_eq!(n("first_crossing(x, 15, falling)"), Some(2.5));
        // No rising crossing of 100 → undefined.
        assert_eq!(n("first_crossing(x, 100, rising)"), None);
        // last_crossing (any direction) of 15 is the falling one at 2.5.
        assert_eq!(n("last_crossing(x, 15)"), Some(2.5));
    }

    #[test]
    fn type_errors_are_hard_errors() {
        let signals = SignalSet::new();
        let ctx = ctx_with(0.0, 1.0, &signals);
        // boolean where a number is expected
        assert!(Expr::parse("true + 1").unwrap().eval(&ctx).is_err());
        // number where a boolean is expected
        assert!(Expr::parse("1 and 2").unwrap().eval(&ctx).is_err());
        // unknown signal
        assert!(Expr::parse("mean(nope)").unwrap().eval(&ctx).is_err());
    }

    #[test]
    fn parse_errors_are_reported() {
        assert!(Expr::parse("1 +").is_err());
        assert!(Expr::parse("mean(").is_err());
        assert!(Expr::parse("bogus(x)").is_err());
        assert!(Expr::parse("1 = 2").is_err()); // must be ==
        assert!(Expr::parse("(1 + 2").is_err());
    }

    #[test]
    fn signals_collects_referenced_names_in_order() {
        let e = Expr::parse("mean(f0) > mean(f0, file) and max(intensity) > -20").unwrap();
        assert_eq!(e.signals(), vec!["f0".to_string(), "intensity".to_string()]);
        // crossing threshold sub-expression is also scanned
        let e = Expr::parse("first_crossing(intensity, mean(energy))").unwrap();
        assert_eq!(
            e.signals(),
            vec!["intensity".to_string(), "energy".to_string()]
        );
    }
}
