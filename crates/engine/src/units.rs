//! Lightweight typed-unit newtypes for the measurement path.
//!
//! The "no bare numbers" substrate from the clinical-regulatory entry,
//! kept deliberately light: thin wrappers that make a quantity's unit
//! part of the type (so a signature reads `Hertz`, not `f32`) without
//! the ceremony of a full dimensional-analysis crate. They unwrap to
//! the underlying primitive at the Python / NumPy boundary, so the
//! language bindings keep returning plain floats.
//!
//! Scope at A2: frequency ([`Hertz`]) and level ([`Decibels`]) — the
//! quantities most prone to unit confusion and most load-bearing for
//! the clinical measures. [`Seconds`] and [`Ratio`] are defined for the
//! cluster-B measures (perturbation, ratios) to consume; note that
//! time across the rest of the engine stays a bare `f64` seconds (the
//! universal time type — wrapping it everywhere buys little).

use std::fmt;

/// Implements the shared surface for an `f32`-backed unit newtype:
/// `new` / `value`, `Display`, and the common derives.
macro_rules! unit_f32 {
    ($(#[$meta:meta])* $name:ident, $suffix:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name(f32);

        impl $name {
            #[doc = concat!("Wraps a bare `f32` as [`", stringify!($name), "`].")]
            pub const fn new(value: f32) -> Self {
                Self(value)
            }
            /// The underlying value, for arithmetic or the FFI boundary.
            pub const fn value(self) -> f32 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{} {}", self.0, $suffix)
            }
        }
    };
}

unit_f32!(
    /// A frequency in hertz (f0, formant centre frequencies, bandwidths).
    Hertz,
    "Hz"
);
unit_f32!(
    /// A level in decibels. Carries no reference on its own — dB-FS
    /// (re digital full scale) and dB-SPL (re 20 µPa) are distinguished
    /// by where the value comes from, documented per measure.
    Decibels,
    "dB"
);
unit_f32!(
    /// A dimensionless ratio (e.g. voicing strength, jitter as a
    /// fraction). Display has no unit suffix beyond a bare number.
    Ratio,
    ""
);

/// A duration in seconds, `f64`-backed to match the engine's universal
/// time representation.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Seconds(f64);

impl Seconds {
    /// Wraps a bare `f64` as [`Seconds`].
    pub const fn new(value: f64) -> Self {
        Self(value)
    }
    /// The underlying value.
    pub const fn value(self) -> f64 {
        self.0
    }
}

impl fmt::Display for Seconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} s", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_unwrap_roundtrips() {
        assert_eq!(Hertz::new(440.0).value(), 440.0);
        assert_eq!(Decibels::new(-3.0).value(), -3.0);
        assert_eq!(Ratio::new(0.5).value(), 0.5);
        assert_eq!(Seconds::new(1.25).value(), 1.25);
    }

    #[test]
    fn display_carries_unit() {
        assert_eq!(Hertz::new(120.0).to_string(), "120 Hz");
        assert_eq!(Decibels::new(-6.0).to_string(), "-6 dB");
        assert_eq!(Seconds::new(0.5).to_string(), "0.5 s");
    }

    #[test]
    fn ordering_works_for_thresholds() {
        assert!(Hertz::new(80.0) < Hertz::new(300.0));
        assert!(Decibels::new(-3.0) > Decibels::new(-6.0));
    }
}
