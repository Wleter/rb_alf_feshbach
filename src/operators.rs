use scattering_problems::{
    scattering_solver::quantum::{
        clebsch_gordan::{half_integer::HalfI32, hi32, hu32, wigner_3j, wigner_6j},
        states::{braket::Braket, spins::Spin},
    },
    utility::{AngularPair, p1_factor, spin_phase_factor},
};

#[rustfmt::skip]
pub fn nuclear_electric_quad_int_mel(ang: Braket<AngularPair>, n_tot: Braket<Spin>, i: Braket<Spin>) -> f64 {
    if ang.bra.l == ang.ket.l && i.bra.s == i.ket.s && n_tot.bra.ms + i.bra.ms == n_tot.ket.ms + i.ket.ms { 
        let factors = 0.25 * p1_factor(n_tot.bra.s) * p1_factor(n_tot.ket.s)
            * p1_factor(ang.bra.n) * p1_factor(ang.ket.n);
        let phase = spin_phase_factor(i.bra) * spin_phase_factor(n_tot.ket)
            * (-1f64).powi(2 + (n_tot.bra.s + ang.bra.l + ang.bra.n + ang.bra.n).double_value() as i32 / 2);
        
        let wigner_divide = wigner_3j(i.bra.s, hu32!(2), i.bra.s, 
                                    -HalfI32::from(i.bra.s), hi32!(0), HalfI32::from(i.bra.s));
        if wigner_divide == 0. {
            return 0.
        }

        let wigners = wigner_6j(ang.ket.n, n_tot.ket.s, ang.bra.l, 
                                n_tot.bra.s, ang.bra.n, hu32!(2))
            * wigner_3j(ang.bra.n, hu32!(2), ang.ket.n, 
                        hi32!(0), hi32!(0), hi32!(0))
            / wigner_divide
            * wigner_3j(n_tot.bra.s, hu32!(2), n_tot.ket.s, 
                        -n_tot.bra.ms, n_tot.bra.ms - n_tot.ket.ms, n_tot.ket.ms)
            * wigner_3j(i.bra.s, hu32!(2), i.bra.s, 
                        -i.bra.ms, i.bra.ms - i.ket.ms, i.ket.ms);

        factors * phase * wigners
    } else {
        0.
    }
}
