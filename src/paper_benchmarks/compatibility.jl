const N8_KC = author_inflation.N8_KC
const N8_BEST_X = author_inflation.N8_BEST_X
const N8_TAU = author_inflation.N8_TAU
const N8_Q = author_inflation.N8_Q
const N8_Q_TRAJECTORY = author_inflation.N8_Q_TRAJECTORY
const N8_TAU_TRAJECTORY = author_inflation.N8_TAU_TRAJECTORY
const N8_K_RAW = author_inflation.N8_K_RAW
const N5_Q = author_inflation.N5_Q
const N5_QDOTTAU = author_inflation.N5_QDOTTAU
const N5_VERTICES = author_inflation.N5_VERTICES
const N5_VOLUME = author_inflation.N5_VOLUME
const N5_DIVISOR_VOLUMES = author_inflation.N5_DIVISOR_VOLUMES
const N5_K_RAW = author_inflation.N5_K_RAW
const N5_LIGHT_DIRECTION = author_inflation.N5_LIGHT_DIRECTION

const n5_geometry = author_inflation.n5_geometry
const n5_kinetic_matrix = author_inflation.n5_kinetic_matrix
const n5_light_direction = author_inflation.n5_light_direction
const n8_author_trajectory = author_inflation.n8_author_trajectory
const n5_hilltop_efolds = author_inflation.n5_hilltop_efolds
const reference_efolds = author_inflation.reference_efolds

"""Scientific namespace for the fixed poly-102 inflation model."""
const poly102_inflation = author_inflation

# Canonical scientific names; the historical names above remain supported.
const n8_local_hilltop_efolds = n8_hilltop_efolds
const n8_physical_gradient_flow = author_inflation.n8_physical_gradient_flow
const n8_hilltop_normal_form_efolds = author_inflation.n8_hilltop_normal_form_efolds
const n8_hilltop_normal_form = author_inflation.n8_hilltop_normal_form
const n8_efold_gradient_flow = author_inflation.n8_efold_gradient_flow
const n5_hilltop_normal_form_efolds = author_inflation.n5_hilltop_normal_form_efolds
const benchmark_efold_targets = author_inflation.benchmark_efold_targets

n8_hilltop_probe(args...; kwargs...) =
    author_inflation.n8_hilltop_probe(args...; kwargs...)
n8_slow_roll_trajectory(args...; kwargs...) =
    author_inflation.n8_slow_roll_trajectory(args...; kwargs...)

function _n8_symmetric_power(matrix::AbstractMatrix{<:Real}, power::Real)
    eigensystem = eigen(Symmetric(Float64.(matrix)))
    minimum(eigensystem.values) > 0 ||
        throw(ArgumentError("kinetic metric must be positive definite"))
    eigensystem.vectors * Diagonal(eigensystem.values .^ power) *
        eigensystem.vectors'
end
function n8_coordinate_maps(k::Real)
    metric = Matrix(n8_kinetic_matrix(k))
    (; metric, raw_to_canonical=_n8_symmetric_power(metric, 1 / 2),
       canonical_to_raw=_n8_symmetric_power(metric, -1 / 2))
end

function n8_mass_eigenbasis(k::Real=N8_KC;
        theta::Union{Nothing,AbstractVector{<:Real}}=nothing,
        basis_k::Real=k)
    basis_theta = theta === nothing ? n8_hilltop(basis_k).theta : Float64.(theta)
    length(basis_theta) == 8 ||
        throw(DimensionMismatch("the N=8 benchmark needs eight coordinates"))
    maps = n8_coordinate_maps(basis_k)
    derivatives = n8_potential_derivatives(basis_theta, basis_k)
    canonical_hessian = maps.canonical_to_raw' * derivatives.hessian *
        maps.canonical_to_raw
    eigensystem = eigen(Symmetric(canonical_hessian))
    raw_eigenvectors = maps.canonical_to_raw * eigensystem.vectors
    (; basis=:mass_eigenbasis, theta=basis_theta, k=Float64(basis_k),
       metric=maps.metric, hessian_theta=derivatives.hessian,
       canonical_hessian, eigenvalues=eigensystem.values,
       canonical_eigenvectors=eigensystem.vectors, raw_eigenvectors,
       orthonormality_residual=norm(raw_eigenvectors' * maps.metric *
           raw_eigenvectors - I, Inf),
       generalized_residual=opnorm(derivatives.hessian * raw_eigenvectors -
           maps.metric * raw_eigenvectors * Diagonal(eigensystem.values)))
end

function n8_unstable_direction(k::Real=N8_KC;
        mode::Symbol=:most_negative, basis::Symbol=:canonical_hessian,
        basis_theta::Union{Nothing,AbstractVector{<:Real}}=nothing,
        basis_k::Real=N8_KC)
    basis in (:canonical_hessian, :mass_eigenbasis) ||
        throw(ArgumentError("unsupported basis: $basis"))
    mass_basis = n8_mass_eigenbasis(k; theta=basis_theta, basis_k=basis_k)
    index = mode === :smallest_abs ? argmin(abs.(mass_basis.eigenvalues)) :
        mode === :most_negative ? argmin(mass_basis.eigenvalues) :
        throw(ArgumentError("unsupported direction mode: $mode"))
    (; raw=mass_basis.raw_eigenvectors[:, index],
       canonical=mass_basis.canonical_eigenvectors[:, index],
       eigenvalues=mass_basis.eigenvalues, index, metric=mass_basis.metric,
       basis, basis_theta=mass_basis.theta, basis_k=mass_basis.k,
       raw_eigenvectors=mass_basis.raw_eigenvectors,
       canonical_eigenvectors=mass_basis.canonical_eigenvectors,
       hessian_theta=mass_basis.hessian_theta,
       generalized_residual=mass_basis.generalized_residual,
       orthonormality_residual=mass_basis.orthonormality_residual)
end

function n8_basis_directions(k::Real=N8_KC)
    unstable = n8_unstable_direction(k; basis=:canonical_hessian)
    mass = n8_unstable_direction(k; basis=:mass_eigenbasis)
    kinetic = eigen(Symmetric(unstable.metric))
    directions = [unstable.metric * kinetic.vectors[:, i] for i in axes(kinetic.vectors, 2)]
    directions = [direction / sqrt(dot(direction, unstable.metric * direction))
        for direction in directions]
    draft_index = argmax(abs.([
        dot(direction, unstable.metric * unstable.raw) for direction in directions]))
    named = (
        A_draft_kinetic=directions[draft_index],
        B_package_current=unstable.raw,
        C_canonical_hessian=unstable.raw,
        D_raw_coordinate=normalize([1.0, zeros(7)...]),
        E_mass_eigenbasis=mass.raw,
    )
    overlap = [dot(left, unstable.metric * right) for left in named, right in named]
    (; directions=named, overlap, metric=unstable.metric,
       metric_eigenvalues=kinetic.values, draft_kinetic_index=draft_index,
       unstable, canonical_hessian=unstable, mass_eigenbasis=mass,
       equivalent_mass_direction=abs(dot(unstable.raw,
           unstable.metric * mass.raw)))
end
