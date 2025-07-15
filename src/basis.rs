use faer::Mat;
use scattering_problems::{
    FieldScatteringProblem, IndexBasisDescription, ScatteringProblem,
    abm::{HifiProblemBuilder, utility::diagonalize},
    alkali_rotor_atom::ParityBlock,
    angular_block::{AngularBlock, AngularBlocks},
    scattering_solver::{
        boundary::Asymptotic,
        potentials::{
            composite_potential::Composite,
            dispersion_potential::Dispersion,
            masked_potential::MaskedPotential,
            multi_diag_potential::Diagonal,
            pair_potential::PairPotential,
            potential::{MatPotential, SimplePotential},
        },
        quantum::{
            cast_variant,
            clebsch_gordan::half_integer::{HalfI32, HalfU32},
            operator_diagonal_mel, operator_mel,
            states::{
                States, StatesBasis,
                spins::{Spin, SpinOperators, get_spin_basis},
                state::{StateBasis, into_variant},
            },
            units::{Au, Energy},
        },
        utility::AngMomentum,
    },
    utility::{AngularPair, create_angular_pairs, percival_coef_tram_mel, spin_rot_tram_mel},
};

use crate::operators::nuclear_electric_quad_int_mel;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Basis {
    Angular(AngularPair),
    NTot(Spin),
    AtomS(Spin),
    AtomI(Spin),
    RotorI1(Spin),
    RotorI2(Spin),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BasisRecipe {
    pub l_max: u32,
    pub n_max: u32,
    pub n_tot_max: u32,
    pub parity: ParityBlock,
    pub tot_m_projection: HalfI32,
}

pub struct SystemParams {
    pub hifi_atom: HifiProblemBuilder,
    pub rotor_i_12: (HalfU32, HalfU32),

    pub rot_const: Energy<Au>,
    pub centr_distortion: Energy<Au>,
    pub gamma_rotor1: Energy<Au>,
    pub gamma_rotor2: Energy<Au>,
    pub spin_rot1: Energy<Au>,
    pub spin_rot2: Energy<Au>,

    /// eqQ term
    pub el_quad: Energy<Au>,
    pub nuclear_spin_spin: Energy<Au>,
}

#[derive(Clone)]
pub struct SystemProblemBuilder<P, V>
where
    P: SimplePotential,
    V: SimplePotential,
{
    pub potential: Vec<(u32, P)>,
    pub aniso_hifi: [Vec<(u32, V)>; 3],
}

impl<P, V> SystemProblemBuilder<P, V>
where
    P: SimplePotential,
    V: SimplePotential,
{
    pub fn build(self, params: &SystemParams, basis_recipe: &BasisRecipe) -> SystemProblem<P, V> {
        use Basis::*;

        let ordered_basis = self.basis(params, basis_recipe);

        assert!(ordered_basis.is_sorted_by_key(|s| cast_variant!(s[0], Angular).l));

        let angular_block_basis = (0..=basis_recipe.l_max)
            .map(|l| {
                let red_basis = ordered_basis
                    .iter()
                    .filter(|s| matches!(s[0], Angular(ang_curr) if ang_curr.l == l))
                    .cloned()
                    .collect::<StatesBasis<Basis>>();

                (l, red_basis)
            })
            .filter(|(_, b)| !b.is_empty())
            .collect::<Vec<(u32, StatesBasis<Basis>)>>();

        let rot_const = params.rot_const.to_au();
        let centr_distortion = params.centr_distortion.to_au();
        let gamma_e = params.hifi_atom.gamma_e;
        let spin_rot1 = params.spin_rot1.to_au();
        let spin_rot2 = params.spin_rot2.to_au();
        let nuclear_spin_spin = params.nuclear_spin_spin.to_au();
        let el_quad = params.el_quad.to_au();
        let gamma_i1 = params.gamma_rotor1.to_au();
        let gamma_i2 = params.gamma_rotor2.to_au();

        let angular_blocks = angular_block_basis
            .iter()
            .map(|(l, basis)| {
                let h_rot = operator_diagonal_mel!(&basis, |[ang: Angular]| {
                    let n2 = ang.n.value() * (ang.n.value() + 1.0);
                    rot_const * n2 - centr_distortion * n2 * n2
                });

                let mut hifi = Mat::zeros(basis.len(), basis.len());
                if let Some(atom_hifi) = params.hifi_atom.a_hifi {
                    hifi += operator_mel!(&basis, |[s: AtomS, i: AtomI]| {
                        atom_hifi * SpinOperators::dot(s, i)
                    })
                    .as_ref();
                }

                let mut zeeman_prop = operator_diagonal_mel!(&basis,
                    |[s_atom: AtomS, i1_rotor: RotorI1, i2_rotor: RotorI2]| {
                        -gamma_e * s_atom.ms.value()
                            - gamma_i1 * i1_rotor.ms.value()
                            - gamma_i2 * i2_rotor.ms.value()
                    }
                )
                .into_backed();
                if let Some(gamma_i) = params.hifi_atom.gamma_i {
                    zeeman_prop += operator_diagonal_mel!(&basis, |[i: AtomI]| {
                        -gamma_i * i.ms.value()
                    })
                    .as_ref()
                }

                let spin_rot1 = operator_mel!(&basis,
                    |[ang: Angular, n_tot: NTot, i1: RotorI1]| {
                        spin_rot1 * spin_rot_tram_mel(ang, n_tot, i1)
                    }
                );

                let spin_rot2 = operator_mel!(&basis,
                    |[ang: Angular, n_tot: NTot, i2: RotorI2]| {
                        spin_rot2 * spin_rot_tram_mel(ang, n_tot, i2)
                    }
                );

                let spin_spin = operator_mel!(&basis, |[i1: RotorI1, i2: RotorI2]| {
                    nuclear_spin_spin * SpinOperators::dot(i1, i2)
                });

                let el_quad = operator_mel!(&basis, |[ang: Angular, n_tot: NTot, i1: RotorI1]| {
                    el_quad * nuclear_electric_quad_int_mel(ang, n_tot, i1)
                });

                let field_inv = vec![
                    hifi,
                    h_rot.into_backed(),
                    spin_rot1.into_backed(),
                    spin_rot2.into_backed(),
                    spin_spin.into_backed(),
                    el_quad.into_backed(),
                ];

                let field_prop = vec![zeeman_prop];

                AngularBlock::new(AngMomentum(*l), field_inv, field_prop)
            })
            .collect();

        let potentials = self
            .potential
            .into_iter()
            .map(|(lambda, pot)| {
                let masking_singlet = operator_mel!(&ordered_basis,
                    |[ang: Angular, n_tot: NTot]| {
                        percival_coef_tram_mel(lambda, ang, n_tot)
                    }
                );

                (pot, masking_singlet.into_backed())
            })
            .collect();

        let [aniso_hifi_atom, aniso_hifi_rotor1, aniso_hifi_rotor2] = self.aniso_hifi;

        let mut aniso_hifi: Vec<(V, Mat<f64>)> = Vec::with_capacity(
            aniso_hifi_atom.len() + aniso_hifi_rotor1.len() + aniso_hifi_rotor2.len(),
        );
        aniso_hifi.extend(aniso_hifi_atom.into_iter().map(|(lambda, pot)| {
            let masking_singlet = operator_mel!(&ordered_basis,
                |[ang: Angular, n_tot: NTot, s_atom: AtomS, i_atom: AtomI]| {
                    SpinOperators::dot(s_atom, i_atom)
                        * percival_coef_tram_mel(lambda, ang, n_tot)
                }
            );

            (pot, masking_singlet.into_backed())
        }));

        aniso_hifi.extend(aniso_hifi_rotor1.into_iter().map(|(lambda, pot)| {
            let masking_singlet = operator_mel!(&ordered_basis,
                |[ang: Angular, n_tot: NTot, s_atom: AtomS, i1: RotorI1]| {
                    SpinOperators::dot(s_atom, i1)
                        * percival_coef_tram_mel(lambda, ang, n_tot)
                }
            );

            (pot, masking_singlet.into_backed())
        }));

        aniso_hifi.extend(aniso_hifi_rotor2.into_iter().map(|(lambda, pot)| {
            let masking_singlet = operator_mel!(&ordered_basis,
                |[ang: Angular, n_tot: NTot, s_atom: AtomS, i2: RotorI2]| {
                    SpinOperators::dot(s_atom, i2)
                        * percival_coef_tram_mel(lambda, ang, n_tot)
                }
            );

            (pot, masking_singlet.into_backed())
        }));

        SystemProblem {
            angular_blocks: AngularBlocks(angular_blocks),
            basis: ordered_basis,
            potentials,
            aniso_hifi,
        }
    }

    fn basis(&self, params: &SystemParams, basis_recipe: &BasisRecipe) -> StatesBasis<Basis> {
        use Basis::*;

        let l_max = basis_recipe.l_max;
        let n_max = basis_recipe.n_max;
        let n_tot_max = basis_recipe.n_tot_max;
        let parity = basis_recipe.parity;

        let angular_states = create_angular_pairs(l_max, n_max, n_tot_max, parity);
        let angular_states = into_variant(angular_states, Angular);

        let total_angular = (0..=n_tot_max)
            .flat_map(|n_tot| into_variant(get_spin_basis(n_tot.into()), NTot))
            .collect();

        let s_atom = into_variant(get_spin_basis(params.hifi_atom.s), AtomS);
        let i_atom = into_variant(get_spin_basis(params.hifi_atom.i), AtomI);
        let i_rotor1 = into_variant(get_spin_basis(params.rotor_i_12.0), RotorI1);
        let i_rotor2 = into_variant(get_spin_basis(params.rotor_i_12.1), RotorI2);

        let mut states = States::default();
        states
            .push_state(StateBasis::new(angular_states))
            .push_state(StateBasis::new(total_angular))
            .push_state(StateBasis::new(s_atom))
            .push_state(StateBasis::new(i_atom))
            .push_state(StateBasis::new(i_rotor1))
            .push_state(StateBasis::new(i_rotor2));

        let mut basis: StatesBasis<Basis> = states
            .iter_elements()
            .filter(|b| {
                let ang = cast_variant!(b[0], Angular);
                let m_n_tot = cast_variant!(b[1], NTot);
                let s_atom = cast_variant!(b[2], AtomS);
                let i_atom = cast_variant!(b[3], AtomI);
                let i1_rotor = cast_variant!(b[4], RotorI1);
                let i2_rotor = cast_variant!(b[5], RotorI2);

                m_n_tot.ms + i1_rotor.ms + i2_rotor.ms + s_atom.ms + i_atom.ms
                    == basis_recipe.tot_m_projection
                    && (ang.l + ang.n) >= m_n_tot.s
                    && (ang.l + m_n_tot.s) >= ang.n
                    && (ang.n + m_n_tot.s) >= ang.l
            })
            .collect();

        basis.sort_by_key(|b| cast_variant!(b[0], Angular).l);

        basis
    }
}

pub struct SystemProblem<P, V> {
    pub basis: StatesBasis<Basis>,
    pub angular_blocks: AngularBlocks,
    pub potentials: Vec<(P, Mat<f64>)>,
    pub aniso_hifi: Vec<(V, Mat<f64>)>,
}

impl<P, V> FieldScatteringProblem<IndexBasisDescription> for SystemProblem<P, V>
where
    P: SimplePotential + Clone,
    V: SimplePotential + Clone,
{
    fn scattering_for(
        &self,
        mag_field: f64,
    ) -> ScatteringProblem<impl MatPotential, IndexBasisDescription> {
        let (energies, states) = self.angular_blocks.diagonalize(mag_field);

        let energy_levels = energies.iter().map(|e| Dispersion::new(*e, 0)).collect();
        let energy_levels = Diagonal::from_vec(energy_levels);

        let potentials = self
            .potentials
            .iter()
            .map(|(p, m)| {
                let masking = states.transpose() * m * states.as_ref();

                MaskedPotential::new(p.clone(), masking)
            })
            .collect();
        let potentials = Composite::from_vec(potentials);

        let aniso_hifi = self
            .aniso_hifi
            .iter()
            .map(|(p, m)| {
                let masking = states.transpose() * m * states.as_ref();

                MaskedPotential::new(p.clone(), masking)
            })
            .collect();
        let aniso_hifi = Composite::from_vec(aniso_hifi);

        let potential = PairPotential::new(potentials, aniso_hifi);
        let full_potential = PairPotential::new(energy_levels, potential);

        let asymptotic = Asymptotic {
            channel_energies: energies,
            centrifugal: self.angular_blocks.angular_states(),
            entrance: 0,
            channel_states: states,
        };

        ScatteringProblem {
            potential: full_potential,
            asymptotic,
            basis_description: IndexBasisDescription,
        }
    }

    fn levels(&self, field: f64, l: Option<u32>) -> (Vec<f64>, Mat<f64>) {
        if let Some(l) = l {
            let block = &self.angular_blocks.0[l as usize];
            let internal = &block.field_inv() + field * &block.field_prop();

            diagonalize(internal.as_ref())
        } else {
            self.angular_blocks.diagonalize(field)
        }
    }
}
