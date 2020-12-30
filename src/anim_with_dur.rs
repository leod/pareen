use std::ops::{Add, Div, Mul, Sub};

use num_traits::{Float, One};

use crate::{Anim, Fun};

/// An `Anim` together with the duration that it is intended to be played for.
///
/// Explicitly carrying the duration around makes it easier to sequentially
/// compose animations in some places.
#[derive(Clone, Debug)]
pub struct AnimWithDur<F: Fun>(pub Anim<F>, pub F::T);

impl<F> Anim<F>
where
    F: Fun,
{
    /// Tag this animation with the duration that it is intended to be played
    /// for.
    ///
    /// Note that using this tagging is completely optional, but it may
    /// make it easier to combine animations sometimes.
    pub fn dur(self, t: F::T) -> AnimWithDur<F> {
        AnimWithDur(self, t)
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::T: Copy + Float,
{
    pub fn scale_to_dur(self, dur: F::T) -> AnimWithDur<impl Fun<T = F::T, V = F::V>> {
        self.scale_time(F::T::one() / dur).dur(dur)
    }
}

impl<F> AnimWithDur<F>
where
    F: Fun,
{
    pub fn dur(self, t: F::T) -> AnimWithDur<F> {
        AnimWithDur(self.0, t)
    }
}

impl<F> AnimWithDur<F>
where
    F: Fun,
    F::T: Copy + PartialOrd + Sub<Output = F::T>,
{
    pub fn seq<G>(self, next: Anim<G>) -> Anim<impl Fun<T = F::T, V = F::V>>
    where
        G: Fun<T = F::T, V = F::V>,
    {
        self.0.seq(self.1, next)
    }
}

impl<F> AnimWithDur<F>
where
    F: Fun,
    F::T: Copy + PartialOrd + Sub<Output = F::T> + Add<Output = F::T>,
{
    pub fn seq_with_dur<G>(self, next: AnimWithDur<G>) -> AnimWithDur<impl Fun<T = F::T, V = F::V>>
    where
        G: Fun<T = F::T, V = F::V>,
    {
        let dur = self.1 + next.1;
        AnimWithDur(self.seq(next.0), dur)
    }
}

impl<F> AnimWithDur<F>
where
    F: Fun,
    F::T: Copy + Float,
{
    pub fn repeat(self) -> Anim<impl Fun<T = F::T, V = F::V>> {
        self.0.repeat(self.1)
    }
}

impl<F> AnimWithDur<F>
where
    F: Fun,
    F::T: Copy + Sub<Output = F::T>,
{
    pub fn backwards(self) -> AnimWithDur<impl Fun<T = F::T, V = F::V>> {
        AnimWithDur(self.0.backwards(self.1), self.1)
    }
}

impl<F> AnimWithDur<F>
where
    F: Fun,
    F::T: Copy + Mul<Output = F::T> + Div<Output = F::T>,
{
    pub fn scale_time(self, t_scale: F::T) -> AnimWithDur<impl Fun<T = F::T, V = F::V>> {
        AnimWithDur(self.0.scale_time(t_scale), self.1 / t_scale)
    }
}

#[macro_export]
macro_rules! seq_with_dur {
    (
        $expr:expr $(,)?
    ) => {
        $expr
    };

    (
        $head:expr,
        $($tail:expr $(,)?)+
    ) => {
        $head.seq_with_dur($crate::seq_with_dur!($($tail,)*))
    }
}
