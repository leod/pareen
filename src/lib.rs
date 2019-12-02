//! Pareen is a small library for *par*ameterized inbetw*een*ing.
//! The intended application is in game programming, where you sometimes have
//! two discrete game states between which you want to transition smoothly
//! for visualization purposes.
//!
//! Pareen gives you tools for composing animations that are parameterized by
//! time (i.e. mappings from time to some animated value) without constantly
//! having to pass around time variables; it hides the plumbing, so that you
//! need to provide time only once: when evaluating the animation.
//!
//! Animations are composed similarly to Rust's iterators, so no memory
//! allocations are necessary.
//! ## Examples
//!
//! ```rust
//! # use assert_approx_eq::assert_approx_eq;
//! // An animation returning a constant value
//! let anim1 = pareen::constant(1.0f64);
//!
//! // Animations can be evaluated at any time
//! let value = anim1.eval(0.5);
//!
//! // Animations can be played in sequence
//! let anim2 = anim1.seq(0.7, pareen::prop(0.25) + 0.5);
//!
//! // Animations can be composed and transformed in various ways
//! let anim3 = anim2
//!     .lerp(pareen::circle().cos())
//!     .scale_min_max(5.0, 10.0)
//!     .backwards(1.0)
//!     .squeeze(3.0, 0.5..=1.0);
//!
//! let anim4 = pareen::cubic(&[1.0, 2.0, 3.0, 4.0]) - anim3;
//!
//! let value = anim4.eval(1.0);
//! assert_approx_eq!(value, 0.0);
//! ```

use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, RangeInclusive, Sub};

use num_traits::{Float, FloatConst, Num, One};

#[cfg(feature = "easer")]
use easer::functions::Easing;

/// A `Fun` represents anything that maps from some type `T` to another
/// type `V`.
///
/// `T` usually stands for time and `V` for some value that is parameterized by
/// time.
///
/// ## Implementation details
/// The only reason that we define this trait instead of just using `Fn(T) -> V`
/// is so that the library works in stable rust. Having this type allows us to
/// implement e.g. `std::ops::Add` for [`Anim<F>`](struct.Anim.html) where
/// `F: Fun`. Without this trait, it becomes difficult (impossible?) to provide
/// a name for `Add::Output`, unless you have the unstable feature
/// `type_alias_impl_trait` or `fn_traits`.
///
/// In contrast to `std::ops::FnOnce`, both input _and_ output are associated
/// types of `Fun`. The main reason is that this makes types smaller for the
/// user of the library. I have not observed any downsides to this yet.
pub trait Fun {
    /// The function's input type. Usually time.
    type T;

    /// The function's output type.
    type V;

    /// Evaluate the function at time `t`.
    fn eval(&self, t: Self::T) -> Self::V;
}

/// `Anim` is the main type provided by pareen. It is a wrapper around any type
/// implementing [`Fun`](trait.Fun.html).
///
/// `Anim` provides methods that transform or compose animations, allowing
/// complex animations to be created out of simple pieces.
#[derive(Clone, Debug)]
pub struct Anim<F>(pub F);

impl<F> Anim<F>
where
    F: Fun,
{
    /// Evaluate the animation at time `t`.
    pub fn eval(&self, t: F::T) -> F::V {
        self.0.eval(t)
    }

    /// Transform an animation so that it applies a given function to its
    /// values.
    ///
    /// # Example
    ///
    /// Turn `(2.0 * t)` into `(2.0 * t).sqrt() + 2.0 * t`:
    /// ```
    /// # use assert_approx_eq::assert_approx_eq;
    /// let anim = pareen::prop(2.0f32).map(|value| value.sqrt() + value);
    ///
    /// assert_approx_eq!(anim.eval(1.0), 2.0f32.sqrt() + 2.0);
    /// ```
    pub fn map<W>(self, f: impl Fn(F::V) -> W) -> Anim<impl Fun<T = F::T, V = W>> {
        self.map_anim(fun(f))
    }

    /// Transform an animation so that it modifies time according to the given
    /// function before evaluating the animation.
    ///
    /// # Example
    /// Run an animation two times slower:
    /// ```
    /// let anim = pareen::cubic(&[1.0, 1.0, 1.0, 1.0]);
    /// let slower_anim = anim.map_time(|t: f32| t / 2.0);
    /// ```
    pub fn map_time<S>(self, f: impl Fn(S) -> F::T) -> Anim<impl Fun<T = S, V = F::V>> {
        self.map_time_anim(fun(f))
    }

    pub fn map_anim<W, G, A>(self, anim: A) -> Anim<impl Fun<T = F::T, V = W>>
    where
        G: Fun<T = F::V, V = W>,
        A: Into<Anim<G>>,
    {
        let anim = anim.into();
        fun(move |t| anim.eval(self.eval(t)))
    }

    pub fn map_time_anim<S, G, A>(self, anim: A) -> Anim<impl Fun<T = S, V = F::V>>
    where
        G: Fun<T = S, V = F::T>,
        A: Into<Anim<G>>,
    {
        let anim = anim.into();
        fun(move |t| self.eval(anim.eval(t)))
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::T: Copy,
{
    /// Combine two animations into one, yielding an animation having pairs as
    /// the values.
    pub fn zip<W, G, A>(self, other: A) -> Anim<impl Fun<T = F::T, V = (F::V, W)>>
    where
        G: Fun<T = F::T, V = W>,
        A: Into<Anim<G>>,
    {
        let other = other.into();

        fun(move |t| (self.eval(t), other.eval(t)))
    }

    pub fn bind<W, G>(self, f: impl Fn(F::V) -> Anim<G>) -> Anim<impl Fun<T = F::T, V = W>>
    where
        G: Fun<T = F::T, V = W>,
    {
        fun(move |t| f(self.eval(t)).eval(t))
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::T: Copy + Sub<Output = F::T>,
{
    /// Shift an animation in time, so that it is moved to the right by `t_delay`.
    pub fn shift_time(self, t_delay: F::T) -> Anim<impl Fun<T = F::T, V = F::V>> {
        self.map_time(move |t| t - t_delay)
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::T: Copy + PartialOrd,
{
    /// Concatenate `self` with another animation in time, using `self` until
    /// time `self_end` (non-inclusive), and then switching to `next`.
    ///
    /// # Examples
    /// Switch from one constant value to another:
    /// ```
    /// # use assert_approx_eq::assert_approx_eq;
    /// let anim = pareen::constant(1.0f32).switch(0.5f32, 2.0);
    ///
    /// assert_approx_eq!(anim.eval(0.0), 1.0);
    /// assert_approx_eq!(anim.eval(0.5), 2.0);
    /// assert_approx_eq!(anim.eval(42.0), 2.0);
    /// ```
    ///
    /// Piecewise combinations of functions:
    /// ```
    /// let cubic_1 = pareen::cubic(&[4.4034, 0.0, -4.5455e-2, 0.0]);
    /// let cubic_2 = pareen::cubic(&[-1.2642e1, 2.0455e1, -8.1364, 1.0909]);
    /// let cubic_3 = pareen::cubic(&[1.6477e1, -4.9432e1, 4.7773e1, -1.3818e1]);
    ///
    /// // Use cubic_1 for [0.0, 0.4), cubic_2 for [0.4, 0.8) and
    /// // cubic_3 for [0.8, ..).
    /// let anim = cubic_1.switch(0.4, cubic_2).switch(0.8, cubic_3);
    /// ```
    pub fn switch<G, A>(self, self_end: F::T, next: A) -> Anim<impl Fun<T = F::T, V = F::V>>
    where
        G: Fun<T = F::T, V = F::V>,
        A: Into<Anim<G>>,
    {
        cond(fun(move |t| t < self_end), self, next)
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::T: Copy + PartialOrd + Sub<Output = F::T>,
{
    /// Play two animations in sequence, first playing `self` until time
    /// `self_end` (non-inclusive), and then switching to `next`. Note that
    /// `next` will see time starting at zero once it plays.
    ///
    /// # Example
    /// Stay at value `5.0` for ten seconds, then increase value proportionally:
    /// ```
    /// # use assert_approx_eq::assert_approx_eq;
    /// let anim_1 = pareen::constant(5.0f32);
    /// let anim_2 = pareen::prop(2.0f32) + 5.0;
    /// let anim = anim_1.seq(10.0, anim_2);
    ///
    /// assert_approx_eq!(anim.eval(0.0), 5.0);
    /// assert_approx_eq!(anim.eval(10.0), 5.0);
    /// assert_approx_eq!(anim.eval(11.0), 7.0);
    /// ```
    pub fn seq<G, A>(self, self_end: F::T, next: A) -> Anim<impl Fun<T = F::T, V = F::V>>
    where
        G: Fun<T = F::T, V = F::V>,
        A: Into<Anim<G>>,
    {
        self.switch(self_end, next.into().shift_time(self_end))
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::T: Copy + Sub<Output = F::T>,
{
    /// Play an animation backwards, starting at time `end`.
    ///
    /// # Example
    ///
    /// ```
    /// # use assert_approx_eq::assert_approx_eq;
    /// let anim = pareen::prop(2.0f32).backwards(1.0);
    ///
    /// assert_approx_eq!(anim.eval(0.0f32), 2.0);
    /// assert_approx_eq!(anim.eval(1.0f32), 0.0);
    /// ```
    pub fn backwards(self, end: F::T) -> Anim<impl Fun<T = F::T, V = F::V>> {
        fun(move |t| self.eval(end - t))
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::T: Copy,
    F::V: Copy + Num,
{
    /// Given animation values in `[0.0 .. 1.0]`, this function transforms the
    /// values so that they are in `[min .. max]`.
    ///
    /// # Example
    /// ```
    /// # use assert_approx_eq::assert_approx_eq;
    /// let min = -3.0f32;
    /// let max = 10.0;
    /// let anim = pareen::id().scale_min_max(min, max);
    ///
    /// assert_approx_eq!(anim.eval(0.0f32), min);
    /// assert_approx_eq!(anim.eval(1.0f32), max);
    /// ```
    pub fn scale_min_max(self, min: F::V, max: F::V) -> Anim<impl Fun<T = F::T, V = F::V>> {
        self * (max - min) + min
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::V: Float,
{
    /// Apply `Float::sin` to the animation values.
    pub fn sin(self) -> Anim<impl Fun<T = F::T, V = F::V>> {
        self.map(Float::sin)
    }

    /// Apply `Float::cos` to the animation values.
    pub fn cos(self) -> Anim<impl Fun<T = F::T, V = F::V>> {
        self.map(Float::cos)
    }

    /// Apply `Float::abs` to the animation values.
    pub fn abs(self) -> Anim<impl Fun<T = F::T, V = F::V>> {
        self.map(Float::abs)
    }

    /// Apply `Float::powf` to the animation values.
    pub fn powf(self, e: F::V) -> Anim<impl Fun<T = F::T, V = F::V>> {
        self.map(move |v| v.powf(e))
    }

    /// Apply `Float::powi` to the animation values.
    pub fn powi(self, n: i32) -> Anim<impl Fun<T = F::T, V = F::V>> {
        self.map(move |v| v.powi(n))
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::T: Copy + Float,
    F::V: Copy,
{
    /// Transform an animation in time, so that its time `[0 .. 1]` is shifted
    /// and scaled into the given `range`.
    ///
    /// In other words, this function can both delay and speed up or slow down a
    /// given animation.
    ///
    /// For time inputs outside the `range`, the `default` value is returned.
    ///
    /// # Example
    ///
    /// Go from zero to 2π in half a second:
    /// ```
    /// # use assert_approx_eq::assert_approx_eq;
    /// // From zero to 2π in one second
    /// let angle = pareen::circle();
    ///
    /// // From zero to 2π from time 0.5 to 1.0
    /// let anim = angle.squeeze(42.0f32, 0.5..=1.0);
    ///
    /// assert_approx_eq!(anim.eval(0.0f32), 42.0);
    /// assert_approx_eq!(anim.eval(0.5), 0.0);
    /// assert_approx_eq!(anim.eval(1.0), std::f32::consts::PI * 2.0);
    /// assert_approx_eq!(anim.eval(1.1), 42.0);
    /// ```
    pub fn squeeze(
        self,
        default: F::V,
        range: RangeInclusive<F::T>,
    ) -> Anim<impl Fun<T = F::T, V = F::V>> {
        let time_shift = *range.start();
        let time_scale = F::T::one() / (*range.end() - *range.start());

        cond(
            move |t| range.contains(&t),
            self.map_time(move |t| (t - time_shift) * time_scale),
            default,
        )
    }
}

impl<W, F> Anim<F>
where
    F: Fun,
    F::T: Copy + Mul<W, Output = W>,
    F::V: Copy + Add<W, Output = F::V> + Sub<Output = W>,
{
    /// Linearly interpolate between two animations, starting at time zero and
    /// finishing at time one.
    ///
    /// # Examples
    ///
    /// Linearly interpolate between two constant values:
    /// ```
    /// # use assert_approx_eq::assert_approx_eq;
    /// let anim = pareen::constant(5.0f32).lerp(10.0);
    ///
    /// assert_approx_eq!(anim.eval(0.0f32), 5.0);
    /// assert_approx_eq!(anim.eval(0.5), 7.5);
    /// assert_approx_eq!(anim.eval(1.0), 10.0);
    /// assert_approx_eq!(anim.eval(2.0), 15.0);
    /// ```
    ///
    /// It is also possible to linearly interpolate between two non-constant
    /// animations:
    /// ```
    /// let anim = pareen::circle().sin().lerp(pareen::circle().cos());
    /// let value: f32 = anim.eval(0.5f32);
    /// ```
    pub fn lerp<G, A>(self, other: A) -> Anim<impl Fun<T = F::T, V = F::V>>
    where
        G: Fun<T = F::T, V = F::V>,
        A: Into<Anim<G>>,
    {
        let other = other.into();

        fun(move |t| {
            let a = self.eval(t);
            let b = other.eval(t);

            let delta = b - a;

            a + t * delta
        })
    }
}

#[cfg(feature = "easer")]
impl<V, F> Anim<F>
where
    V: Float,
    F: Fun<T = V, V = V>,
{
    fn seq_ease<G, H, A>(
        self,
        self_end: V,
        ease: impl Fn(V, V, V) -> Anim<G>,
        ease_duration: V,
        next: A,
    ) -> Anim<impl Fun<T = V, V = V>>
    where
        G: Fun<T = V, V = V>,
        H: Fun<T = V, V = V>,
        A: Into<Anim<H>>,
    {
        let next = next.into();

        let ease_start_value = self.eval(self_end);
        let ease_end_value = next.eval(V::zero());
        let ease_delta = ease_end_value - ease_start_value;
        let ease = ease(ease_start_value, ease_delta, ease_duration);

        self.seq(self_end, ease).seq(self_end + ease_duration, next)
    }

    /// Play two animations in sequence, transitioning between them with an
    /// easing-in function from
    /// [`easer`](https://docs.rs/easer/0.2.1/easer/index.html).
    ///
    /// This is only available when enabling the `easer` feature for `pareen`.
    ///
    /// The values of `self` at `self_end` and of `next` at time zero are used
    /// to determine the parameters of the easing function.
    ///
    /// Note that, as with [`seq`](struct.Anim.html#method.seq), the `next`
    /// animation will see time starting at zero once it plays.
    ///
    /// # Arguments
    ///
    /// * `self_end` - Time at which the `self` animation is to stop.
    /// * `_easing` - A struct implementing
    ///     [`easer::functions::Easing`](https://docs.rs/easer/0.2.1/easer/functions/trait.Easing.html).
    ///     This determines the easing function that will be used for the
    ///     transition. It is passed as a parameter here to simplify type
    ///     inference.
    /// * `ease_duration` - The amount of time to use for transitioning to `next`.
    /// * `next` - The animation to play after transitioning.
    ///
    /// # Example
    ///
    /// See [`seq_ease_in_out`](struct.Anim.html#method.seq_ease_in_out) for an example.
    pub fn seq_ease_in<E, G, A>(
        self,
        self_end: V,
        _easing: E,
        ease_duration: V,
        next: A,
    ) -> Anim<impl Fun<T = V, V = V>>
    where
        E: Easing<V>,
        G: Fun<T = V, V = V>,
        A: Into<Anim<G>>,
    {
        self.seq_ease(self_end, ease_in::<E, V>, ease_duration, next)
    }

    /// Play two animations in sequence, transitioning between them with an
    /// easing-out function from
    /// [`easer`](https://docs.rs/easer/0.2.1/easer/index.html).
    ///
    /// This is only available when enabling the `easer` feature for `pareen`.
    ///
    /// The values of `self` at `self_end` and of `next` at time zero are used
    /// to determine the parameters of the easing function.
    ///
    /// Note that, as with [`seq`](struct.Anim.html#method.seq), the `next`
    /// animation will see time starting at zero once it plays.
    ///
    /// # Arguments
    ///
    /// * `self_end` - Time at which the `self` animation is to stop.
    /// * `_easing` - A struct implementing
    ///     [`easer::functions::Easing`](https://docs.rs/easer/0.2.1/easer/functions/trait.Easing.html).
    ///     This determines the easing function that will be used for the
    ///     transition. It is passed as a parameter here to simplify type
    ///     inference.
    /// * `ease_duration` - The amount of time to use for transitioning to `next`.
    /// * `next` - The animation to play after transitioning.
    ///
    /// # Example
    ///
    /// See [`seq_ease_in_out`](struct.Anim.html#method.seq_ease_in_out) for an example.
    pub fn seq_ease_out<E, G, A>(
        self,
        self_end: V,
        _: E,
        ease_duration: V,
        next: A,
    ) -> Anim<impl Fun<T = V, V = V>>
    where
        E: Easing<V>,
        G: Fun<T = V, V = V>,
        A: Into<Anim<G>>,
    {
        self.seq_ease(self_end, ease_out::<E, V>, ease_duration, next)
    }

    /// Play two animations in sequence, transitioning between them with an
    /// easing-in-out function from
    /// [`easer`](https://docs.rs/easer/0.2.1/easer/index.html).
    ///
    /// This is only available when enabling the `easer` feature for `pareen`.
    ///
    /// The values of `self` at `self_end` and of `next` at time zero are used
    /// to determine the parameters of the easing function.
    ///
    /// Note that, as with [`seq`](struct.Anim.html#method.seq), the `next`
    /// animation will see time starting at zero once it plays.
    ///
    /// # Arguments
    ///
    /// * `self_end` - Time at which the `self` animation is to stop.
    /// * `_easing` - A struct implementing
    ///     [`easer::functions::Easing`](https://docs.rs/easer/0.2.1/easer/functions/trait.Easing.html).
    ///     This determines the easing function that will be used for the
    ///     transition. It is passed as a parameter here to simplify type
    ///     inference.
    /// * `ease_duration` - The amount of time to use for transitioning to `next`.
    /// * `next` - The animation to play after transitioning.
    ///
    /// # Example
    ///
    /// Play a constant value until time `0.5`, then transition for `0.3`
    /// time units, using a cubic function, into a second animation:
    /// ```
    /// let first_anim = pareen::constant(2.0);
    /// let second_anim = pareen::prop(1.0f32);
    /// let anim = first_anim.seq_ease_in_out(
    ///     0.5,
    ///     easer::functions::Cubic,
    ///     0.3,
    ///     second_anim,
    /// );
    /// ```
    /// The animation will look like this:
    ///
    /// ![plot for seq_ease_in_out](https://raw.githubusercontent.com/leod/pareen/master/images/seq_ease_in_out.png)
    pub fn seq_ease_in_out<E, G, A>(
        self,
        self_end: V,
        _: E,
        ease_duration: V,
        next: A,
    ) -> Anim<impl Fun<T = V, V = V>>
    where
        E: Easing<V>,
        G: Fun<T = V, V = V>,
        A: Into<Anim<G>>,
    {
        self.seq_ease(self_end, ease_in_out::<E, V>, ease_duration, next)
    }
}

impl<V, F> Anim<F>
where
    F: Fun<V = Option<V>>,
    F::T: Copy,
{
    /// Unwrap an animation of optional values.
    ///
    /// At any time, returns the animation value if it is not `None`, or the
    /// given `default` value otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// let anim1 = pareen::constant(Some(42)).unwrap_or(-1);
    /// assert_eq!(anim1.eval(2), 42);
    /// assert_eq!(anim1.eval(3), 42);
    /// ```
    ///
    /// ```
    /// let cond = pareen::fun(|t| t % 2 == 0);
    /// let anim1 = pareen::cond(cond, Some(42), None).unwrap_or(-1);
    /// assert_eq!(anim1.eval(2), 42);
    /// assert_eq!(anim1.eval(3), -1);
    /// ```
    pub fn unwrap_or<G, A>(self, default: A) -> Anim<impl Fun<T = F::T, V = V>>
    where
        G: Fun<T = F::T, V = V>,
        A: Into<Anim<G>>,
    {
        self.zip(default.into())
            .map(|(v, default)| v.unwrap_or(default))
    }

    /// Applies a function to the contained value (if any), or returns the
    /// provided default (if not).
    ///
    /// Note that the function `f` itself returns an animation.
    ///
    /// # Example
    ///
    /// Animate a player's position offset if it is moving:
    /// ```
    /// # use assert_approx_eq::assert_approx_eq;
    /// fn my_offset_anim(
    ///     move_dir: Option<f32>,
    /// ) -> pareen::Anim<impl pareen::Fun<T = f32, V = f32>> {
    ///     let move_speed = 2.0f32;
    ///
    ///     pareen::constant(move_dir).map_or(
    ///         0.0,
    ///         move |move_dir| pareen::prop(move_dir) * move_speed,
    ///     )
    /// }
    ///
    /// let move_anim = my_offset_anim(Some(1.0));
    /// let stay_anim = my_offset_anim(None);
    ///
    /// assert_approx_eq!(move_anim.eval(0.5), 1.0);
    /// assert_approx_eq!(stay_anim.eval(0.5), 0.0);
    /// ```
    pub fn map_or<W, G, H, A>(
        self,
        default: A,
        f: impl Fn(V) -> Anim<H>,
    ) -> Anim<impl Fun<T = F::T, V = W>>
    where
        G: Fun<T = F::T, V = W>,
        H: Fun<T = F::T, V = W>,
        A: Into<Anim<G>>,
    {
        let default = default.into();

        //self.bind(move |v| v.map_or(default, f))

        fun(move |t| {
            self.eval(t)
                .map_or_else(|| default.eval(t), |v| f(v).eval(t))
        })
    }
}

/// Turn any function `Fn(T) -> V` into an [`Anim`](struct.Anim.html).
///
/// # Example
/// ```
/// # use assert_approx_eq::assert_approx_eq;
/// fn my_crazy_function(t: f32) -> f32 {
///     42.0 / t
/// }
///
/// let anim = pareen::fun(my_crazy_function);
///
/// assert_approx_eq!(anim.eval(1.0), 42.0);
/// assert_approx_eq!(anim.eval(2.0), 21.0);
/// ```
pub fn fun<T, V>(f: impl Fn(T) -> V) -> Anim<impl Fun<T = T, V = V>> {
    From::from(f)
}

/// A constant animation, always returning the same value.
///
/// # Example
/// ```
/// # use assert_approx_eq::assert_approx_eq;
/// let anim = pareen::constant(1.0f32);
///
/// assert_approx_eq!(anim.eval(-10000.0f32), 1.0);
/// assert_approx_eq!(anim.eval(0.0), 1.0);
/// assert_approx_eq!(anim.eval(42.0), 1.0);
/// ```
pub fn constant<T, V: Copy>(c: V) -> Anim<impl Fun<T = T, V = V>> {
    fun(move |_| c)
}

/// An animation that returns a value proportional to time.
///
/// # Example
///
/// Scale time with a factor of three:
/// ```
/// # use assert_approx_eq::assert_approx_eq;
/// let anim = pareen::prop(3.0f32);
///
/// assert_approx_eq!(anim.eval(0.0f32), 0.0);
/// assert_approx_eq!(anim.eval(3.0), 9.0);
/// ```
pub fn prop<T, V, W>(m: V) -> Anim<impl Fun<T = T, V = W>>
where
    V: Copy + Mul<Output = W> + From<T>,
{
    fun(move |t| m * From::from(t))
}

/// An animation that returns time as its value.
///
/// This is the same as [`prop`](fn.prop.html) with a factor of one.
///
/// # Examples
/// ```
/// let anim = pareen::id::<isize, isize>();
///
/// assert_eq!(anim.eval(-100), -100);
/// assert_eq!(anim.eval(0), 0);
/// assert_eq!(anim.eval(100), 100);
/// ```
/// ```
/// # use assert_approx_eq::assert_approx_eq;
/// let anim = pareen::id() * 3.0f32 + 4.0;
///
/// assert_approx_eq!(anim.eval(0.0), 4.0);
/// assert_approx_eq!(anim.eval(100.0), 304.0);
/// ```
pub fn id<T, V>() -> Anim<impl Fun<T = T, V = V>>
where
    V: From<T>,
{
    fun(From::from)
}

/// Proportionally increase value from zero to 2π.
pub fn circle<T, V>() -> Anim<impl Fun<T = T, V = V>>
where
    T: Float,
    V: Float + FloatConst + From<T>,
{
    prop(V::PI() * (V::one() + V::one()))
}

/// Proportionally increase value from zero to π.
pub fn half_circle<T, V>() -> Anim<impl Fun<T = T, V = V>>
where
    T: Float,
    V: Float + FloatConst + From<T>,
{
    prop(V::PI())
}

/// Proportionally increase value from zero to π/2.
pub fn quarter_circle<T, V>() -> Anim<impl Fun<T = T, V = V>>
where
    T: Float,
    V: Float + FloatConst + From<T>,
{
    prop(V::PI() * (V::one() / (V::one() + V::one())))
}

/// Return the value of one of two animations depending on a condition.
///
/// This allows returning animations of different types conditionally.
///
/// Note that the condition `cond` may either be a value `true` and `false`, or
/// it may itself be a dynamic animation of type `bool`.
///
/// For dynamic conditions, in many cases it suffices to use either
/// [`Anim::switch`](struct.Anim.html#method.switch) or
/// [`Anim::seq`](struct.Anim.html#method.seq) instead of this function.
///
/// # Examples
/// ## Constant conditions
///
/// The following example does _not_ compile, because the branches have
/// different types:
/// ```compile_fail
/// let cond = true;
/// let anim = if cond { pareen::constant(1) } else { pareen::id() };
/// ```
///
/// However, this does compile:
/// ```
/// let cond = true;
/// let anim = pareen::cond(cond, 1, pareen::id());
///
/// assert_eq!(anim.eval(2), 1);
/// ```
///
/// ## Dynamic conditions
///
/// ```
/// let cond = pareen::fun(|t| t * t <= 4);
/// let anim_1 = 1;
/// let anim_2 = pareen::id();
/// let anim = pareen::cond(cond, anim_1, anim_2);
///
/// assert_eq!(anim.eval(1), 1); // 1 * 1 <= 4
/// assert_eq!(anim.eval(2), 1); // 2 * 2 <= 4
/// assert_eq!(anim.eval(3), 3); // 3 * 3 > 4
/// ```
pub fn cond<T, V, F, G, H, Cond, A, B>(cond: Cond, a: A, b: B) -> Anim<impl Fun<T = T, V = V>>
where
    T: Copy,
    F: Fun<T = T, V = bool>,
    G: Fun<T = T, V = V>,
    H: Fun<T = T, V = V>,
    Cond: Into<Anim<F>>,
    A: Into<Anim<G>>,
    B: Into<Anim<H>>,
{
    let cond = cond.into();
    let a = a.into();
    let b = b.into();

    fun(move |t| if cond.eval(t) { a.eval(t) } else { b.eval(t) })
}

/// Linearly interpolate between two animations, starting at time zero and
/// finishing at time one.
///
/// This is a wrapper around [`Anim::lerp`](struct.Anim.html#method.lerp) for
/// convenience, allowing automatic conversion into [`Anim`](struct.Anim.html)
/// for both `a` and `b`.
///
/// # Example
///
/// Linearly interpolate between two constant values:
///
/// ```
/// # use assert_approx_eq::assert_approx_eq;
/// let anim = pareen::lerp(5.0f32, 10.0);
///
/// assert_approx_eq!(anim.eval(0.0f32), 5.0);
/// assert_approx_eq!(anim.eval(0.5), 7.5);
/// assert_approx_eq!(anim.eval(1.0), 10.0);
/// assert_approx_eq!(anim.eval(2.0), 15.0);
/// ```
pub fn lerp<T, V, W, F, G, A, B>(a: A, b: B) -> Anim<impl Fun<T = T, V = V>>
where
    T: Copy + Mul<W, Output = W>,
    V: Copy + Add<W, Output = V> + Sub<Output = W>,
    F: Fun<T = T, V = V>,
    G: Fun<T = T, V = V>,
    A: Into<Anim<F>>,
    B: Into<Anim<G>>,
{
    a.into().lerp(b.into())
}

/// Evaluate a cubic polynomial in time.
pub fn cubic<T>(w: &[T; 4]) -> Anim<impl Fun<T = T, V = T> + '_>
where
    T: Float,
{
    fun(move |t| {
        let t2 = t * t;
        let t3 = t2 * t;

        w[0] * t3 + w[1] * t2 + w[2] * t + w[3]
    })
}

/// Build an animation that depends on matching some expression.
///
/// Importantly, this macro allows returning animations of a different type in
/// each match arm, which is not possible with a normal `match` expression.
///
/// # Example
/// ```
/// # use assert_approx_eq::assert_approx_eq;
/// enum MyPlayerState {
///     Standing,
///     Running,
///     Jumping,
/// }
///
/// fn my_anim(state: MyPlayerState) -> pareen::Anim<impl pareen::Fun<T = f64, V = f64>> {
///     pareen::anim_match!(state;
///         MyPlayerState::Standing => pareen::constant(0.0),
///         MyPlayerState::Running => pareen::prop(1.0),
///         MyPlayerState::Jumping => pareen::id().powi(2),
///     )
/// }
///
/// assert_approx_eq!(my_anim(MyPlayerState::Standing).eval(2.0), 0.0);
/// assert_approx_eq!(my_anim(MyPlayerState::Running).eval(2.0), 2.0);
/// assert_approx_eq!(my_anim(MyPlayerState::Jumping).eval(2.0), 4.0);
/// ```
#[macro_export]
macro_rules! anim_match {
    (
        $expr:expr;
        $($pat:pat => $value:expr $(,)?)*
    ) => {
        $crate::fun(move |t| match $expr {
            $(
                $pat => ($crate::Anim::from($value)).eval(t),
            )*
        })
    }
}

/// Integrate an easing-in function from the
/// [`easer`](https://docs.rs/easer/0.2.1/easer/index.html) library.
///
/// This is only available when enabling the `easer` feature for `pareen`.
///
/// # Arguments
///
/// * `start` - The start value for the easing function.
/// * `delta` - The change in the value from beginning to end time.
/// * `duration` - The total time between beginning and end.
///
/// # See also
/// Documentation for [`easer::functions::Easing`](https://docs.rs/easer/0.2.1/easer/functions/trait.Easing.html).
#[cfg(feature = "easer")]
pub fn ease_in<E, V>(start: V, delta: V, duration: V) -> Anim<impl Fun<T = V, V = V>>
where
    V: Float,
    E: Easing<V>,
{
    fun(move |t| E::ease_in(t, start, delta, duration))
}

/// Integrate an easing-out function from the
/// [`easer`](https://docs.rs/easer/0.2.1/easer/index.html) library.
///
/// This is only available when enabling the `easer` feature for `pareen`.
///
/// # Arguments
///
/// * `start` - The start value for the easing function.
/// * `delta` - The change in the value from beginning to end time.
/// * `duration` - The total time between beginning and end.
///
/// # See also
/// Documentation for [`easer::functions::Easing`](https://docs.rs/easer/0.2.1/easer/functions/trait.Easing.html).
#[cfg(feature = "easer")]
pub fn ease_out<E, V>(start: V, delta: V, duration: V) -> Anim<impl Fun<T = V, V = V>>
where
    V: Float,
    E: Easing<V>,
{
    fun(move |t| E::ease_out(t, start, delta, duration))
}

/// Integrate an easing-in-out function from the
/// [`easer`](https://docs.rs/easer/0.2.1/easer/index.html) library.
///
/// This is only available when enabling the `easer` feature for `pareen`.
///
/// # Arguments
///
/// * `start` - The start value for the easing function.
/// * `delta` - The change in the value from beginning to end time.
/// * `duration` - The total time between beginning and end.
///
/// # See also
/// Documentation for [`easer::functions::Easing`](https://docs.rs/easer/0.2.1/easer/functions/trait.Easing.html).
#[cfg(feature = "easer")]
pub fn ease_in_out<E, V>(start: V, delta: V, duration: V) -> Anim<impl Fun<T = V, V = V>>
where
    V: Float,
    E: Easing<V>,
{
    fun(move |t| E::ease_in_out(t, start, delta, duration))
}

impl<T, V, F> From<F> for Anim<WrapFn<T, V, F>>
where
    F: Fn(T) -> V,
{
    fn from(f: F) -> Self {
        Anim(WrapFn(f, PhantomData))
    }
}

struct WrapFn<T, V, F: Fn(T) -> V>(F, PhantomData<(T, V)>);

impl<T, V, F> Fun for WrapFn<T, V, F>
where
    F: Fn(T) -> V,
{
    type T = T;
    type V = V;

    fn eval(&self, t: T) -> V {
        self.0(t)
    }
}

impl<F, G> Add<Anim<G>> for Anim<F>
where
    F: Fun,
    G: Fun<T = F::T>,
    F::V: Add<G::V>,
{
    type Output = Anim<AddClosure<F, G>>;

    fn add(self, rhs: Anim<G>) -> Self::Output {
        Anim(AddClosure(self.0, rhs.0))
    }
}

impl<V, F> Add<V> for Anim<F>
where
    V: Copy,
    F: Fun<V = V>,
{
    type Output = Anim<AddClosure<F, ConstantClosure<F::T, F::V>>>;

    fn add(self, rhs: F::V) -> Self::Output {
        Anim(AddClosure(self.0, ConstantClosure::from(rhs)))
    }
}

impl<F, G> Sub<Anim<G>> for Anim<F>
where
    F: Fun,
    G: Fun<T = F::T>,
    F::V: Sub<G::V>,
{
    type Output = Anim<AddClosure<F, NegClosure<G>>>;

    fn sub(self, rhs: Anim<G>) -> Self::Output {
        Anim(AddClosure(self.0, NegClosure(rhs.0)))
    }
}

impl<F, G> Mul<Anim<G>> for Anim<F>
where
    F: Fun,
    F::T: Copy,
    G: Fun<T = F::T>,
    F::V: Mul<G::V>,
{
    type Output = Anim<MulClosure<F, G>>;

    fn mul(self, rhs: Anim<G>) -> Self::Output {
        Anim(MulClosure(self.0, rhs.0))
    }
}

impl<V, F> Mul<V> for Anim<F>
where
    V: Copy,
    F: Fun<V = V>,
    F::T: Copy,
{
    type Output = Anim<MulClosure<F, ConstantClosure<F::T, F::V>>>;

    fn mul(self, rhs: F::V) -> Self::Output {
        Anim(MulClosure(self.0, ConstantClosure::from(rhs)))
    }
}

impl<V, F> Neg for Anim<F>
where
    V: Copy,
    F: Fun<V = V>,
{
    type Output = Anim<NegClosure<F>>;

    fn neg(self) -> Self::Output {
        Anim(NegClosure(self.0))
    }
}

#[doc(hidden)]
pub struct ConstantClosure<T, V>(V, PhantomData<T>);

impl<T, V> Fun for ConstantClosure<T, V>
where
    T: Copy,
    V: Copy,
{
    type T = T;
    type V = V;

    fn eval(&self, _: T) -> V {
        self.0
    }
}

impl<T, V> From<V> for ConstantClosure<T, V>
where
    V: Copy,
{
    fn from(v: V) -> Self {
        ConstantClosure(v, PhantomData)
    }
}

impl<T, V> From<V> for Anim<ConstantClosure<T, V>>
where
    V: Copy,
{
    fn from(v: V) -> Self {
        Anim(ConstantClosure::from(v))
    }
}

#[doc(hidden)]
pub struct AddClosure<F, G>(F, G);

impl<F, G> Fun for AddClosure<F, G>
where
    F: Fun,
    F::T: Copy,
    G: Fun<T = F::T>,
    F::V: Add<G::V>,
{
    type T = F::T;
    type V = <F::V as Add<G::V>>::Output;

    fn eval(&self, t: F::T) -> Self::V {
        self.0.eval(t) + self.1.eval(t)
    }
}

#[doc(hidden)]
pub struct MulClosure<F, G>(F, G);

impl<F, G> Fun for MulClosure<F, G>
where
    F: Fun,
    F::T: Copy,
    G: Fun<T = F::T>,
    F::V: Mul<G::V>,
{
    type T = F::T;
    type V = <F::V as Mul<G::V>>::Output;

    fn eval(&self, t: F::T) -> Self::V {
        self.0.eval(t) * self.1.eval(t)
    }
}

#[doc(hidden)]
pub struct NegClosure<F>(F);

impl<F> Fun for NegClosure<F>
where
    F: Fun,
    F::V: Neg,
{
    type T = F::T;
    type V = <F::V as Neg>::Output;

    fn eval(&self, t: F::T) -> Self::V {
        -self.0.eval(t)
    }
}
