//! Pareen is a small library for *par*ameterized inbetw*een*ing.
//!
//! Pareen allows you to compose animations that are parameterized by time, i.e.
//! mappings from time to some animated value. Its intended application is in
//! game programming, where you sometimes have two discrete game states between
//! which you want to transition smoothly. Pareen gives you tools for combining
//! animations without constantly having to pass around time variables; it
//! hides the plumbing, so that you need to provide time only once: when
//! evaluating the animation.

use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, RangeInclusive, Sub};

use num_traits::{Float, FloatConst, Num, One, Zero};

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

/// `Anim` is the main type provided by pareen. It is a simple wrapper around
/// any type implementing [`Fun`](trait.Fun.html).
#[derive(Clone, Debug)]
pub struct Anim<F>(F);

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
    /// let anim = pareen::proportional(2.0f32).map(|value| value.sqrt() + value);
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
    F::T: Copy + PartialOrd,
{
    /// Concatenate `self` with another animation in time, using `self` until
    /// time `self_end` (non-inclusive), and then switching to `next`.
    ///
    /// # Example
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
    /// // Use `cubic_1` for [0.0, 0.4), `cubic_2` for [0.4, 0.8)` and
    /// // `cubic_3` for `[0.8, ..)`.
    /// let anim = cubic_1.switch(0.4, cubic_2).switch(0.8, cubic_3);
    /// ```
    pub fn switch<G, A>(self, self_end: F::T, next: A) -> Anim<impl Fun<T = F::T, V = F::V>>
    where
        G: Fun<T = F::T, V = F::V>,
        A: Into<Anim<G>>,
    {
        cond_t(fun(move |t| t < self_end), self, next)
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::T: Copy + Sub<Output = F::T>,
{
    pub fn shift_time(self, t_add: F::T) -> Anim<impl Fun<T = F::T, V = F::V>> {
        self.map_time(move |t| t - t_add)
    }
}

impl<F> Anim<F>
where
    F: Fun,
    F::T: Copy + PartialOrd + Sub<Output = F::T>,
{
    /// Play two animations in sequence, first playing `self` until time
    /// `self_end`, and then switching to `next`. Note that `next` will see
    /// time starting at zero once it plays.
    ///
    /// # Example
    /// Stay at value `5.0` for ten seconds, then increase value proportionally:
    /// ```
    /// let anim_1 = pareen::constant(5.0).seq(10.0, pareen::proportional(2.0) + 5.0);
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
    pub fn squeeze(
        self,
        default: F::V,
        range: RangeInclusive<F::T>,
    ) -> Anim<impl Fun<T = F::T, V = F::V>> {
        let time_shift = *range.start();
        let time_scale = F::T::one() / (*range.end() - *range.start());

        cond_t(
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
    /// It is also possible to linearly interpolate between two animations:
    /// ```
    /// let anim = pareen::full_circle().sin().lerp(pareen::full_circle().cos());
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

impl<V, F> Anim<F>
where
    F: Fun<V = Option<V>>,
    F::T: Copy,
{
    pub fn unwrap_or<G, A>(self, default: A) -> Anim<impl Fun<T = F::T, V = V>>
    where
        G: Fun<T = F::T, V = V>,
        A: Into<Anim<G>>,
    {
        self.zip(default.into())
            .map(|(v, default)| v.unwrap_or(default))
    }

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
/// fn my_crazy_function(t: f32) -> f32 {
///     42.0
/// }
///
/// let anim = pareen::fun(my_crazy_function);
/// ```
pub fn fun<T, V>(f: impl Fn(T) -> V) -> Anim<impl Fun<T = T, V = V>> {
    From::from(f)
}

pub fn constant<T, V: Copy>(c: V) -> Anim<impl Fun<T = T, V = V>> {
    fun(move |_| c)
}

pub fn one<T, V: Copy + One>() -> Anim<impl Fun<T = T, V = V>> {
    constant(V::one())
}

pub fn zero<T, V: Copy + Zero>() -> Anim<impl Fun<T = T, V = V>> {
    constant(V::zero())
}

pub fn proportional<T, V, W>(m: V) -> Anim<impl Fun<T = T, V = W>>
where
    V: Copy + Mul<Output = W> + From<T>,
{
    fun(move |t| m * From::from(t))
}

pub fn id<T, V>() -> Anim<impl Fun<T = T, V = V>>
where
    V: From<T>,
{
    fun(move |t| From::from(t))
}

pub fn full_circle<T, V>() -> Anim<impl Fun<T = T, V = V>>
where
    T: Float,
    V: Float + FloatConst + From<T>,
{
    proportional(V::PI() * (V::one() + V::one()))
}

pub fn half_circle<T, V>() -> Anim<impl Fun<T = T, V = V>>
where
    T: Float,
    V: Float + FloatConst + From<T>,
{
    proportional(V::PI())
}

pub fn quarter_circle<T, V>() -> Anim<impl Fun<T = T, V = V>>
where
    T: Float,
    V: Float + FloatConst + From<T>,
{
    proportional(V::PI() * (V::one() / (V::one() + V::one())))
}

pub fn cond_t<T, V, F, G, H, Cond, A, B>(cond: Cond, a: A, b: B) -> Anim<impl Fun<T = T, V = V>>
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

pub fn cond<T, V, F, G, A, B>(cond: bool, a: A, b: B) -> Anim<impl Fun<T = T, V = V>>
where
    T: Copy,
    F: Fun<T = T, V = V>,
    G: Fun<T = T, V = V>,
    A: Into<Anim<F>>,
    B: Into<Anim<G>>,
{
    cond_t(fun(move |_| cond), a, b)
}

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

#[macro_export]
macro_rules! anim_match {
    (
        $expr:expr;
        $($pat:pat => $value:expr $(,)?)*
    ) => {
        $crate::util::anim::fun(move |t| match $expr {
            $(
                $pat => ($crate::util::anim::Anim::from($value)).eval(t),
            )*
        })
    }
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
