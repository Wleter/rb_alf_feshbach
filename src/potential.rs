use std::{f64::consts::PI, fs::File, io::{BufRead, BufReader}};

use faer::Mat;
use gauss_quad::GaussLegendre;
use scattering_problems::{potential_interpolation::{interpolate_potentials, PotentialArray, TransitionedPotential}, scattering_solver::{potentials::{composite_potential::Composite, dispersion_potential::Dispersion, potential::SimplePotential}, quantum::{units::{CmInv, Unit}, utility::legendre_polynomials}}};


pub fn get_potential() -> Vec<(u32, impl SimplePotential + Clone)> {
    let data = load_grid_data();

    let potentials = interpolate_potentials(&data, 3);

    let mut potentials_far = Vec::new();
    for _ in 0..potentials.len() {
        potentials_far.push(Composite::new(Dispersion::new(0., 0)));
    }

    potentials_far[0]
        .add_potential(Dispersion::new(-1096.4, -6));
    potentials_far[2]
        .add_potential(Dispersion::new(-73.8, -6));

    let transition = |r| {
        if r <= 40. {
            1.
        } else if r >= 50. {
            0.
        } else {
            0.5 * (1. + f64::cos(PI * (r - 40.) / 10.))
        }
    };

    potentials
        .into_iter()
        .zip(potentials_far.into_iter())
        .map(|((lambda, near), far)| {
            let combined = TransitionedPotential::new(near, far, transition);

            (lambda, combined)
        })
        .collect()
}

pub fn load_grid_data() -> PotentialArray {
    let filename = "potential_data/X_2Ap_Rb+AlF.txt";
    let mut path = std::env::current_dir().unwrap();
    path.push(filename);
    let f = File::open(&path).expect(&format!(
        "couldn't find potential in provided path {path:?}"
    ));
    let f = BufReader::new(f);

    let angle_no = 11;
    let r_no = 56;

    let mut line_iter = f.lines().skip(1);
    let mut rs = Vec::with_capacity(r_no);
    let mut polars = Vec::with_capacity(angle_no);
    let mut data = Mat::zeros(r_no, angle_no);

    for polar_index in 0..angle_no {
        for r_index in 0..r_no {
            let line = line_iter.next().expect("End of file").unwrap();
            let splitted: Vec<&str> = line.trim().split_whitespace().collect();

            let r = splitted[0].parse::<f64>().unwrap();
            let polar = splitted[1].parse::<f64>().unwrap() * PI / 180.;
            let pot = splitted[2].parse::<f64>().unwrap() * CmInv::TO_AU_MUL;

            if r_index >= rs.len() {
                rs.push(r)
            } else {
                assert_eq!(r, rs[r_index])
            }
            if polar_index >= polars.len() {
                polars.push(polar)
            } else {
                assert_eq!(polar, polars[polar_index])
            }

            data[(r_index, polar_index)] = pot
        }
    }

    let gauss = GaussLegendre::new(angle_no).unwrap();
    let weights = gauss.weights();
    let polynomials: Vec<Vec<f64>> = polars
        .iter()
        .map(|x| legendre_polynomials(angle_no as u32 - 1, x.cos()))
        .collect();

    let mut potentials = Vec::new();
    for lambda in 0..angle_no {
        let values_potential = data.row_iter()
            .map(|x| {
                let pot_dec: f64 = weights.clone()
                    .into_iter()
                    .zip(x.iter())
                    .zip(polynomials.iter().map(|ps| ps[lambda as usize]))
                    .map(|((w, pot), leg)| (lambda as f64 + 0.5) * w * leg * pot)
                    .sum();

                pot_dec
            })
            .collect::<Vec<f64>>();
        potentials.push((lambda as u32, values_potential));
    }

    PotentialArray {
        distances: rs,
        potentials,
    }
}