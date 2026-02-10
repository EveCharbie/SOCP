# Stochastic Optimal Control Problem Transcriptions for biomechanics
This repository comprises the transcriptions of stochastic optimal control problems presented in 
TODO(add link to the paper) and their applications to biomechanical examples presented in TODO(add link to the paper). 
The code is meant to be as easy as possible to read side by side with the associated mathematical formulations.
It is meant to be reused to implement other examples and to be used as a reference for the implementation of the transcriptions.
If you have any questions, do not hesitate to contact [eve.charbie@gmail.com](eve.charbie@gmail.com).

# Cite
The transcriptions presented in this repository are obviously inspired by previous implementations.
If you use or, in your turn get inspired by, this repository please cite the paper associated with this repository:
...TODO 

and the paper from which the transcriptions were inspired too:
- DirectCollocationPolynomial (deterministic): "Michaud, B., Bailly, F., Charbonneau, E., Ceglia, A., Sanchez, L., & Begon, M. (2022). Bioptim, a python framework for musculoskeletal optimal control in biomechanics. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 53(1), 321-332." [pyomeca/Bioptim](https://github.com/pyomeca/bioptim/blob/master/bioptim/dynamics/integrator.py)
- DirectCollocationPolynomial x MeanAndCovariance: "Gillis, J., & Diehl, M. (2013, December). A positive definiteness preserving discretization method for lyapunov differential equations. In 52nd IEEE Conference on Decision and Control (pp. 7759-7764). IEEE."
- DirectCollocationTrapezoidal x MeanAndCovariance: "Van Wouwe, T., Ting, L. H., & De Groote, F. (2022). An approximate stochastic optimal control framework to simulate nonlinear neuro-musculoskeletal models in the presence of noise. PLoS computational biology, 18(6), e1009338." [tomvanwouwe1992/SOC_Paper](https://github.com/tomvanwouwe1992/SOC_Paper/tree/main/SOC_PAPER_REACHING/Integrator)
- DirectMultipleShooting (deterministic): "Michaud, B., Bailly, F., Charbonneau, E., Ceglia, A., Sanchez, L., & Begon, M. (2022). Bioptim, a python framework for musculoskeletal optimal control in biomechanics. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 53(1), 321-332." [pyomeca/Bioptim](https://github.com/pyomeca/bioptim/blob/master/bioptim/dynamics/integrator.py)
- Variational (deterministic): "Puchaud, P., Dumas, R., & Begon, M. Exploring the Benefits of Variational Integrators with Natural Coordinates: A Pendulum Example." [ipuch/variational_integrator](https://github.com/Ipuch/variational_integrator/blob/master/varint/minimal_variational_integrator.py)
- VariationalPolynomial (deterministic): "Campos, C. M., Ober-Blöbaum, S., & Trélat, E. (2015). High order variational integrators in the optimal control of mechanical systems. arXiv preprint arXiv:1502.00325." [cmcampos-xyz/paper-2013-hovi-ocms](https://github.com/cmcampos-xyz/paper-2013-hovi-ocms/blob/main/varInt.m)
