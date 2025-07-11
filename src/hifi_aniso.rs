use std::{f64::consts::PI, fs::File, io::{BufRead, BufReader}};

use scattering_problems::{potential_interpolation::{interpolate_potentials, PotentialArray, TransitionedPotential}, scattering_solver::{potentials::{composite_potential::Composite, dispersion_potential::Dispersion, potential::SimplePotential}, quantum::units::{MHz, Unit}}};

#[derive(Debug, Clone, Copy)]
pub enum AnisoType {
    Rb,
    Al,
    F
}

pub fn get_aniso_hifi(aniso_type: AnisoType) -> Vec<(u32, impl SimplePotential + Clone)> {
    let data = load_aniso_hifi_data(aniso_type);

    let potentials = interpolate_potentials(&data, 3);

    let mut potentials_far = Vec::new();
    for _ in 0..potentials.len() {
        potentials_far.push(Composite::new(Dispersion::new(0., 0)));
    }

    let transition = |r| {
        if r <= 25.9 {
            1.
        } else if r >= 26.0 {
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

pub fn load_aniso_hifi_data(aniso_type: AnisoType) -> PotentialArray {
    let filename = match aniso_type {
        AnisoType::Rb => "param_iso87.dat",
        AnisoType::Al => "param_iso27.dat",
        AnisoType::F => "param_iso19.dat",
    };

    let mut path = std::env::current_dir().unwrap();
    path.push("potential_data");
    path.push(filename);
    let f = File::open(&path).expect(&format!(
        "couldn't find potential in provided path {path:?}"
    ));
    let f = BufReader::new(f);

    let mut rs = Vec::new();
    let mut data = Vec::new();

    let l_max = 10;
    for l in 0..=l_max {
        data.push((l as u32, Vec::new()))
    }

    for line in f.lines().skip(1) {
        let line = line.unwrap();
        let splitted: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(splitted.len(), l_max + 2, "Inconsistent number of data per line");

        rs.push(splitted[0].parse::<f64>().unwrap());
        for l in 0..=l_max {
                            // -1 factor because of mismatch in angle convention
            data[l].1.push((-1.0f64).powi(l as i32) * splitted[l + 1].parse::<f64>().unwrap() * MHz::TO_AU_MUL);
        }
    }

    if let AnisoType::Rb = aniso_type {
        for l in 0..=l_max {
            let value_inf = *data[l].1.last().unwrap();

            for d in data[l].1.iter_mut() {
                *d -= value_inf
            }
        }
    }

    PotentialArray { distances: rs, potentials: data }
}