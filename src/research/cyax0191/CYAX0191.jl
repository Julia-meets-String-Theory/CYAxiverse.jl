"""Isolated CYAX-0191 research arm. This module is not included by `CYAxiverse`."""
module CYAX0191

using LinearAlgebra
using SHA

include("geometry.jl")
include("model.jl")
include("numerics.jl")
include("policy.jl")

export CanonicalIntersectionTensor, GeometrySourceIdentity, ConeProvenance,
    GeometryRecord, NativePayloadImporter, import_geometry, dense_intersections,
    calabi_yau_volume, divisor_volumes, divisor_volume_jacobian,
    cone_margins, imported_domain_status, geometry_identity,
    change_divisor_basis, synthetic_geometry_fixture,
    ModelConvention, ModelSwitches, UpliftSpec, KahlerModel, ModelEvaluation,
    source_basis_model, charge_coordinates, charge_metric_contraction,
    MetricAssessment, evaluate_potential, potential, quadratic_phase,
    direct_complex_interference_phase, analytic_axion_gradient,
    DifferentiationBackend, CentralDifferenceBackend, gradient!, hessian!,
    finite_difference_gradient, finite_difference_hessian,
    scaled_stationarity_residual, SearchBackend, SearchCriteria,
    DampedNewtonSearch, SearchResult, search_stationary, Assessment,
    CriticalPointState, critical_point_state, FluctuationBackend,
    GeneralizedEigenBackend, FluctuationReport, fluctuation_analysis,
    exact_axionic_shift_basis,
    ControlReport, ResearchResult, unassessed_controls, ReplayBackend,
    NativeReplayBackend, replay_manifest, change_charge_basis,
    change_model_basis, change_coordinate_basis, FrozenNumericalPolicy,
    FROZEN_GATE_C_POLICIES, frozen_policy, policy_manifest

end
