use gnuplot::{AxesCommon, Color, Figure};

fn main() {
    let mut plots = Plots { plots: Vec::new() };
    plots.add("id", pareen::id());
    plots.add("lerp between 5 and 10", pareen::lerp(5.0, 10.0));
    plots.add(
        "dynamic lerp between sin^2 and cos",
        pareen::circle().sin().powi(2).lerp(pareen::circle().cos()),
    );
    plots.add(
        "switch from 5 to 10 at time=0.7",
        pareen::constant(5.0).switch(0.7, 10.0),
    );

    plots.show_gnuplot();
}

fn sample(
    n: usize,
    max_t: f32,
    anim: pareen::Anim<impl pareen::Fun<T = f32, V = f32>>,
) -> (Vec<f32>, Vec<f32>) {
    let mut ts = Vec::new();
    let mut vs = Vec::new();

    for i in 0..n {
        let time = i as f32 / n as f32 * max_t;
        let value = anim.eval(time);

        ts.push(time);
        vs.push(value);
    }

    (ts, vs)
}

struct Plot {
    name: &'static str,
    ts: Vec<f32>,
    vs: Vec<f32>,
}

struct Plots {
    plots: Vec<Plot>,
}

impl Plots {
    fn add(&mut self, name: &'static str, anim: pareen::Anim<impl pareen::Fun<T = f32, V = f32>>) {
        let (ts, vs) = sample(100, 1.0, anim);

        self.plots.push(Plot { name, ts, vs });
    }

    fn show_gnuplot(&self) {
        let mut figure = Figure::new();

        // Show plots in a square rows/columns layout
        let square_size = (self.plots.len() as f32).sqrt().ceil() as u32;

        for (i, plot) in self.plots.iter().enumerate() {
            figure
                .axes2d()
                .lines(&plot.ts, &plot.vs, &[Color("blue")])
                .set_title(&plot.name, &[])
                .set_x_label("time", &[])
                .set_y_label("value", &[])
                .set_pos_grid(square_size, square_size, i as u32);
        }

        figure.show().unwrap();
    }
}
