# Pareen
Pareen is a small library for *par*ameterized inbetw*een*ing.

Its intended application is in game programming, where you sometimes have
two discrete game states between which you want to transition smoothly
for visualization purposes.

Pareen gives you tools for composing animations that are parameterized by
time (i.e. mappings from time to some animated value) without constantly
having to pass around time variables; it hides the plumbing, so that you
need to provide time only once: when evaluating the animation.

Animations are composed similarly to Rust's iterators, so no memory
allocations are necessary.

## Usage

## Examples
```rust
// An animation returning a constant value
let anim1 = pareen::constant(1.0f64);

// Animations can be evaluated at any time
let value = anim1.eval(0.5);

// Animations can be played in sequence
let anim2 = anim1.seq(0.7, pareen::prop(0.25) + 0.5);

// Animations can be composed and transformed in various ways
let anim3 = anim2
    .lerp(pareen::circle().cos())
    .scale_min_max(5.0, 10.0)
    .backwards(1.0)
    .squeeze(3.0, 0.5..=1.0);

let anim4 = pareen::cubic(&[1.0, 2.0, 3.0, 4.0]) - anim3;

let value = anim4.eval(1.0);
```

There is an example that shows some animations as plots via [RustGnuplot](https://github.com/SiegeLord/RustGnuplot) in [examples/plots.rs]. Given that `gnuplot` has been installed, it can be executed like this:
```bash
cargo run --example plots
```
