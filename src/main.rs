use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressIterator};
use scattering_problems::{abm::{consts::Consts, HifiProblemBuilder}, alkali_rotor_atom::ParityBlock, scattering_solver::{boundary::{Boundary, Direction}, log_derivatives::johnson::{Johnson, JohnsonLogDerivative}, numerovs::LocalWavelengthStepRule, observables::{bound_states::{BoundProblemBuilder, BoundStates, BoundStatesDependence, WaveFunctions}, s_matrix::{ScatteringDependence, ScatteringObservables}}, potentials::potential::{Potential, SimplePotential}, propagator::{CoupledEquation, Propagator}, quantum::{clebsch_gordan::{hi32, hu32}, params::{particle::Particle, particles::Particles}, problem_selector::{get_args, ProblemSelector}, problems_impl, units::{Au, CmInv, Dalton, Energy, EnergyUnit, GHz, Kelvin, MHz, Mass}, utility::linspace}, utility::{save_data, save_serialize, save_spectrum}}, FieldScatteringProblem};

use crate::{basis::{BasisRecipe, SystemProblemBuilder, SystemParams, SystemProblem}, hifi_aniso::{get_aniso_hifi, load_aniso_hifi_data, AnisoType}, potential::{get_potential, load_grid_data}};

use rayon::prelude::*;

pub mod basis;
pub mod operators;
pub mod potential;
pub mod hifi_aniso;

pub fn main() {
    Problem::select(&mut get_args());
}

pub struct Problem;

problems_impl!(Problem, "AlF + Rb problems",
    "potential" => |_| Self::potential(),
    "anisotropic hyperfine" => |_| Self::aniso_hifi(),
    "scattering length" => |_| Self::scattering_length(),
    "bound states" => |_| Self::bound_states(),
    "levels" => |_| Self::levels(),
    "convergence" => |_| Self::convergence(),
    "bound states structureless" => |_| Self::structureless_bounds(),
    "wave function calculation" => |_| Self::wave_function(),
);

impl Problem {
    fn potential() {
        let pes = load_grid_data();

        let mut data = vec![pes.distances.clone()];
        for (_, p) in &pes.potentials {
            data.push(p.clone());
        }

        save_data(
            "potential_dec_data",
            "distances\tpotential_decomposition",
            &data,
        )
        .unwrap();
    
        let distances = linspace(4., 100., 1000);
        let pes = get_potential();

        let mut data = vec![distances.clone()];
        for (_, p) in &pes {
            data.push(distances.iter().map(|&r| p.value(r)).collect());
        }

        save_data(
            "potential_dec_interpolated",
            "distances\tpotential_decomposition",
            &data,
        )
        .unwrap();
    }

    fn aniso_hifi() {
        let aniso_types = [AnisoType::Rb, AnisoType::Al, AnisoType::F];

        for aniso_type in aniso_types {
            let pes = load_aniso_hifi_data(aniso_type);

            let mut data = vec![pes.distances.clone()];
            for (_, p) in &pes.potentials {
                data.push(p.clone());
            }

            save_data(
                &format!("aniso_hifi_{aniso_type:?}_data"),
                "distances\taniso_hifi_decomposition",
                &data,
            )
            .unwrap();
        
            let distances = linspace(4.5, 50., 500);
            let pes = get_aniso_hifi(aniso_type);

            let mut data = vec![distances.clone()];
            for (_, p) in &pes {
                data.push(distances.iter().map(|&r| p.value(r)).collect());
            }

            save_data(
                &format!("aniso_hifi_{aniso_type:?}_interpolated"),
                "distances\taniso_hifi_decomposition",
                &data,
            )
            .unwrap();
        }
    }

    fn scattering_length() {
        let mag_fields = linspace(0., 8000., 80001);
        
        let basis_recipe = BasisRecipe {
            l_max: 0,
            n_max: 0,
            n_tot_max: 0,
            tot_m_projection: hi32!(4),
            parity: ParityBlock::Positive
        };

        let entrance = 0;

        let energy_relative = Energy(1e-7, Kelvin);
        
        let params = get_params();

        let atoms = get_particles(energy_relative);
        let alkali_problem = get_problem(&params, &basis_recipe);

        let start = Instant::now();
        let scatterings = mag_fields
            .par_iter()
            .progress()
            .map(|&mag_field| {
                let mut atoms = atoms.clone();

                let alkali_problem = alkali_problem.scattering_for(mag_field);
                let mut asymptotic = alkali_problem.asymptotic;
                asymptotic.entrance = entrance;
                atoms.insert(asymptotic);
                let potential = &alkali_problem.potential;

                let boundary = Boundary::new_multi_vanishing(4.6, Direction::Outwards, potential.size());
                let step_rule = LocalWavelengthStepRule::new(4e-3, 10., 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = JohnsonLogDerivative::new(eq, boundary, step_rule);

                numerov.propagate_to(1500.);

                numerov.s_matrix().observables()
            })
            .collect::<Vec<ScatteringObservables>>();

        let elapsed = start.elapsed();
        println!("calculated in {:.2} s", elapsed.as_secs_f64());

        let data = ScatteringDependence {
            parameters: mag_fields,
            observables: scatterings,
        };

        let filename = format!("alf_rb_scattering_n_max_{}", basis_recipe.n_max);
        save_serialize(&filename, &data).unwrap()
    }

    fn bound_states() {
        let mag_fields = linspace(0., 8000., 1001);
        let basis_recipe = BasisRecipe {
            l_max: 65,
            n_max: 65,
            n_tot_max: 0,
            tot_m_projection: hi32!(4),
            parity: ParityBlock::Positive
        };
        let energy_range = (Energy(-6., GHz), Energy(0., GHz));
        let err = Energy(0.05, MHz);

        let entrance = 0;
        let energy_relative = Energy(1e-7, Kelvin);

        let params = get_params();
        let atoms = get_particles(energy_relative);
        let alkali_problem = get_problem(&params, &basis_recipe);

        let start = Instant::now();
        let bound_states = mag_fields
            .iter()
            // .par_iter()
            // .progress()
            .map(|&mag_field| {
                let mut atoms = atoms.clone();

                let alkali_problem = alkali_problem.scattering_for(mag_field);
                let mut asymptotic = alkali_problem.asymptotic;
                asymptotic.entrance = entrance;
                atoms.insert(asymptotic);
                let potential = &alkali_problem.potential;

                let bound_problem = BoundProblemBuilder::new(&atoms, potential)
                    .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
                    .with_range(4.6, 20., 400.)
                    .build();

                let bounds = bound_problem
                    .bound_states(energy_range, err)
                    .with_energy_units(GHz);

                for (e, n) in bounds.energies.iter().zip(bounds.nodes.iter()) {
                    println!("{}, {}, {}", mag_field, n, e);
                }

                bounds
            })
            .collect::<Vec<BoundStates>>();

        let elapsed = start.elapsed();
        println!("calculated in {:.2} s", elapsed.as_secs_f64());

        let data = BoundStatesDependence {
            parameters: mag_fields,
            bound_states,
        };

        let filename = format!("alf_rb_bound_states_n_max_{}", basis_recipe.n_max);
        save_serialize(&filename, &data).unwrap()
    }

    fn levels() {
        let mag_fields = linspace(0., 200., 500);
        let basis_recipe = BasisRecipe {
            l_max: 0,
            n_max: 0,
            n_tot_max: 0,
            tot_m_projection: hi32!(4),
            parity: ParityBlock::Positive
        };
        
        let params = get_params();
        let alkali_problem = get_problem(&params, &basis_recipe);

        let start = Instant::now();
        let levels = mag_fields
            .par_iter()
            .progress()
            .map(|&mag_field| {
                let (levels, _) = alkali_problem.levels(mag_field, None);
                
                levels.into_iter().map(|x| Energy(x, Au).to(GHz).value()).collect()
            })
            .collect::<Vec<Vec<f64>>>();

        let elapsed = start.elapsed();
        println!("calculated in {:.2} s", elapsed.as_secs_f64());

        let filename = format!("alf_rb_levels_n_max_{}", basis_recipe.n_max);
        save_spectrum(
            &filename, 
            "field\tlevels",
            &mag_fields,
            &levels
        ).unwrap()
    }

    fn convergence() {
        let values = [1e-4, 1e-3, 4e-3, 1e-2];

        let basis_recipe = BasisRecipe {
            l_max: 10,
            n_max: 10,
            n_tot_max: 0,
            tot_m_projection: hi32!(4),
            parity: ParityBlock::Positive
        };

        let entrance = 0;

        let energy_relative = Energy(1e-7, Kelvin);
        
        let params = get_params();

        let atoms = get_particles(energy_relative);
        let alkali_problem = get_problem(&params, &basis_recipe);

        let start = Instant::now();
        let scatterings = values
            .par_iter()
            .progress()
            .map(|&value| {
                let mut atoms = atoms.clone();

                let alkali_problem = alkali_problem.scattering_for(0.);
                let mut asymptotic = alkali_problem.asymptotic;
                asymptotic.entrance = entrance;
                atoms.insert(asymptotic);
                let potential = &alkali_problem.potential;

                let boundary = Boundary::new_multi_vanishing(4.6, Direction::Outwards, potential.size());
                let step_rule = LocalWavelengthStepRule::new(value, 10., 400.);
                let eq = CoupledEquation::from_particles(potential, &atoms);
                let mut numerov = JohnsonLogDerivative::new(eq, boundary, step_rule);

                numerov.propagate_to(1500.);

                numerov.s_matrix().observables()
            })
            .collect::<Vec<ScatteringObservables>>();

        let elapsed = start.elapsed();
        println!("calculated in {:.2} s", elapsed.as_secs_f64());

        let data = ScatteringDependence {
            parameters: values.into(),
            observables: scatterings,
        };

        let filename = format!("alf_rb_scattering_n_max_{}_convergence_dr", basis_recipe.n_max);
        save_serialize(&filename, &data).unwrap()
    }

    fn structureless_bounds() {
        let basis_recipe = BasisRecipe {
            l_max: 65,
            n_max: 65,
            n_tot_max: 0,
            tot_m_projection: hi32!(0),
            // tot_m_projection: hi32!(5),
            parity: ParityBlock::Positive
        };

        // let params = get_params();
        let params = {
            let hifi_atom = HifiProblemBuilder::new(hu32!(0), hu32!(0));

            SystemParams {
                hifi_atom: hifi_atom,
                rot_const: Energy(0.549992, CmInv).to(Au),
                rotor_i_12: (hu32!(0), hu32!(0)),
                centr_distortion: Energy(1.04072e-6, CmInv).to(Au),
                gamma_rotor1: Energy(0., Au),
                gamma_rotor2: Energy(0., Au),
                spin_rot1: Energy(0., CmInv).to(Au),
                spin_rot2: Energy(0., CmInv).to(Au),
                el_quad: Energy(0., CmInv).to(Au),
                nuclear_spin_spin: Energy(0., CmInv).to(Au),
            }
        };

        let energy_range = (Energy(-100., GHz), Energy(0., GHz));
        let err = Energy(0.05, MHz);

        let entrance = 0;
        let energy_relative = Energy(1e-7, Kelvin);

        let atoms = get_particles(energy_relative);
        let alkali_problem = get_problem(&params, &basis_recipe);

        let start = Instant::now();
        let mut atoms = atoms.clone();

        let alkali_problem = alkali_problem.scattering_for(0.);
        let mut asymptotic = alkali_problem.asymptotic;
        asymptotic.entrance = entrance;
        atoms.insert(asymptotic);
        let potential = &alkali_problem.potential;

        let bound_problem = BoundProblemBuilder::new(&atoms, potential)
            .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
            .with_range(4.6, 20., 400.)
            .build();

        let bounds = bound_problem
            .bound_states(energy_range, err)
            .with_energy_units(GHz);

        let elapsed = start.elapsed();
        println!("calculated in {:.2} s", elapsed.as_secs_f64());

        let data = BoundStatesDependence {
            parameters: vec![0.],
            bound_states: vec![bounds],
        };

        let filename = format!("alf_rb_bound_states_n_max_{}_structureless_distortion", basis_recipe.n_max);
        save_serialize(&filename, &data).unwrap()
    }

    fn wave_function() {
        let basis_recipe = BasisRecipe {
            l_max: 65,
            n_max: 65,
            n_tot_max: 0,
            // tot_m_projection: hi32!(0),
            tot_m_projection: hi32!(4),// no structure -- 5
            parity: ParityBlock::Positive
        };
        let mag_field = 1822.;
        let suffix = "1822_field";

        let energy_range = (Energy(-0.1, GHz), Energy(0., GHz));
        let err = Energy(0.005, MHz);

        let params = get_params();
        // let params = {
        //     let hifi_atom = HifiProblemBuilder::new(hu32!(0), hu32!(0));

        //     SystemParams {
        //         hifi_atom: hifi_atom,
        //         rot_const: Energy(0.549992, CmInv).to(Au),
        //         rotor_i_12: (hu32!(0), hu32!(0)),
        //         centr_distortion: Energy(1.04072e-6, CmInv).to(Au),
        //         gamma_rotor1: Energy(0., Au),
        //         gamma_rotor2: Energy(0., Au),
        //         spin_rot1: Energy(0., CmInv).to(Au),
        //         spin_rot2: Energy(0., CmInv).to(Au),
        //         el_quad: Energy(0., CmInv).to(Au),
        //         nuclear_spin_spin: Energy(0., CmInv).to(Au),
        //     }
        // };


        let entrance = 0;
        let energy_relative = Energy(1e-7, Kelvin);

        let atoms = get_particles(energy_relative);
        let alkali_problem = get_problem(&params, &basis_recipe);

        let start = Instant::now();
        let mut atoms = atoms.clone();

        let alkali_problem = alkali_problem.scattering_for(mag_field);
        let mut asymptotic = alkali_problem.asymptotic;
        asymptotic.entrance = entrance;
        atoms.insert(asymptotic);
        let potential = &alkali_problem.potential;

        let bound_problem = BoundProblemBuilder::new(&atoms, potential)
            .with_propagation(LocalWavelengthStepRule::new(4e-3, 10., 400.), Johnson)
            .with_range(4.6, 20., 1000.)
            .build();

        let bounds = bound_problem
            .bound_states(energy_range, err);

        let waves = bound_problem.bound_waves(&bounds).map(|x| x.normalize()).collect();

        let elapsed = start.elapsed();
        println!("calculated in {:.2} s", elapsed.as_secs_f64());

        let waves = WaveFunctions {
            bounds,
            waves,
        };

        let filename = format!("alf_rb_wavefunction_n_max_{}_{suffix}", basis_recipe.n_max);
        save_serialize(&filename, &waves).unwrap()
    }
}

pub fn get_params() -> SystemParams {
    let hifi_atom = HifiProblemBuilder::new(hu32!(1/2), hu32!(3/2))
        .with_nuclear_magneton(Energy(1.834216 * Consts::NUCLEAR_MAG, Au).to_au())
        .with_hyperfine_coupling(Energy(0.113990, CmInv).to_au());

    SystemParams {
        hifi_atom: hifi_atom,
        rotor_i_12: (hu32!(5/2), hu32!(1/2)),
        rot_const: Energy(0.549992, CmInv).to(Au),
        centr_distortion: Energy(1.04072e-6, CmInv).to(Au),
        gamma_rotor1: Energy(1.45628 * Consts::NUCLEAR_MAG, Au),
        gamma_rotor2: Energy(5.256642 * Consts::NUCLEAR_MAG, Au),
        spin_rot1: Energy(3.46907e-7, CmInv).to(Au),
        spin_rot2: Energy(1.20083e-6, CmInv).to(Au),
        el_quad: Energy(-1.2517326e-3, CmInv).to(Au),
        nuclear_spin_spin: Energy(2.201523e-7, CmInv).to(Au),
    }
}

pub fn get_problem(
    params: &SystemParams,
    basis_recipe: &BasisRecipe,
) -> SystemProblem<
    impl SimplePotential + Clone + use<>,
    impl SimplePotential + Clone + use<>,
> {
    let potential = get_potential();
    let aniso_hifi = [get_aniso_hifi(AnisoType::Rb), get_aniso_hifi(AnisoType::Al), get_aniso_hifi(AnisoType::F)];

    SystemProblemBuilder {
        potential,
        aniso_hifi,
    }.build(params, basis_recipe)
}

pub fn get_particles(energy: Energy<impl EnergyUnit>) -> Particles {
    let rb = Particle::new("Rb78", Mass(86.909180527, Dalton));
    let srf = Particle::new("AlF", Mass(45.97994157, Dalton));

    Particles::new_pair(rb, srf, energy)
}