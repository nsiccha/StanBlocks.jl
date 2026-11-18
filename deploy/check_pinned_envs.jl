# Landing check `pinned-envs` (kb-fleet.toml): every committed Manifest.toml
# agrees with the in-repo packages it develops by `path`.
#
# The deployed app (deploy/gordito/launch.sh, `# kb-resolve: web/app`) starts
# from its committed, pinned Manifest, which the KB instantiates as committed
# and never re-resolves. A Manifest records each developed package's
# dependency list at the time it was resolved, while Julia loads that
# package's source from the path. So a commit that adds a dependency to
# StanBlocks (or PosteriorDBWeb) without re-pinning the env still
# instantiates cleanly, and the app then dies at `using` with `Package
# StanBlocks does not have <Dep> in its dependencies` (snag
# opened-p-stanblo-37c68cc6: StaticArrays, added in v0.2.49, took the app
# down). This check refuses such a commit before it lands.
#
# It compares, for every in-repo `path` entry of every tracked Manifest, the
# package's Project.toml `[deps]`, `[weakdeps]`, `[extensions]` and uuid with
# what the Manifest entry records, and checks that every dependency has a
# Manifest entry of its own. The recorded `version` is not compared: loading
# ignores it, and a version bump alone needs no re-pin.
#
# Run: julia --startup-file=no deploy/check_pinned_envs.jl
using TOML

const ROOT = dirname(@__DIR__)

names_of(x::AbstractVector) = Set{String}(x)
names_of(x::AbstractDict) = Set{String}(keys(x))
names_of(::Nothing) = Set{String}()

ext_map(::Nothing) = Dict{String,Set{String}}()
ext_map(x::AbstractDict) =
    Dict{String,Set{String}}(k => Set{String}(v isa AbstractString ? [v] : v) for (k, v) in x)

function tracked_manifests(root)
    out = read(`git -C $root ls-files -z -- '*Manifest.toml'`, String)
    sort!(filter(f -> basename(f) == "Manifest.toml", split(out, '\0'; keepempty=false)))
end

inside(root, path) = !startswith(relpath(path, root), "..")

function project_file(dir)
    for name in ("JuliaProject.toml", "Project.toml")
        isfile(joinpath(dir, name)) && return joinpath(dir, name)
    end
    nothing
end

describe(s) = isempty(s) ? "none" : join(sort!(collect(s)), ", ")

function check_manifest(root, rel, problems)
    before = length(problems)
    env = dirname(joinpath(root, rel))
    envrel = relpath(env, root)
    manifest = TOML.parsefile(joinpath(root, rel))
    get(manifest, "manifest_format", "1.0") == "2.0" ||
        error("$rel: manifest_format $(get(manifest, "manifest_format", "1.0")) is not supported by this check; expected 2.0")
    entries = get(manifest, "deps", Dict{String,Any}())
    envproject = project_file(env)
    envproject === nothing && error("$rel: no Project.toml beside it")
    for name in names_of(get(TOML.parsefile(envproject), "deps", nothing))
        haskey(entries, name) ||
            push!(problems, "$rel: the env's Project.toml lists $name, but the Manifest has no entry for it")
    end
    checked = String[]
    for (name, list) in entries, entry in list
        haskey(entry, "path") || continue
        pkgdir = normpath(joinpath(env, entry["path"]))
        inside(root, pkgdir) || continue
        where = "$rel: [[deps.$name]] (path = \"$(entry["path"])\")"
        pf = project_file(pkgdir)
        if pf === nothing
            push!(problems, "$where: no Project.toml at $(relpath(pkgdir, root))")
            continue
        end
        project = TOML.parsefile(pf)
        push!(checked, name)
        get(project, "uuid", nothing) == get(entry, "uuid", nothing) ||
            push!(problems, "$where: uuid $(get(entry, "uuid", "missing")) differs from $(relpath(pf, root))'s $(get(project, "uuid", "missing"))")
        for (field, label) in (("deps", "[deps]"), ("weakdeps", "[weakdeps]"))
            want = names_of(get(project, field, nothing))
            have = names_of(get(entry, field, nothing))
            missing_ = setdiff(want, have)
            extra = setdiff(have, want)
            isempty(missing_) ||
                push!(problems, "$where: $(relpath(pf, root)) $label has $(describe(missing_)), which the Manifest entry does not record")
            isempty(extra) ||
                push!(problems, "$where: the Manifest entry records $field $(describe(extra)), which $(relpath(pf, root)) $label no longer has")
        end
        ext_map(get(project, "extensions", nothing)) == ext_map(get(entry, "extensions", nothing)) ||
            push!(problems, "$where: [extensions] differ from $(relpath(pf, root))")
        for dep in names_of(get(project, "deps", nothing))
            haskey(entries, dep) ||
                push!(problems, "$where: dependency $dep has no Manifest entry of its own")
        end
    end
    pkgs = isempty(checked) ? "no in-repo packages" : join(sort!(checked), ", ")
    println("pinned-envs: ", envrel, length(problems) == before ? " agrees with " : " is out of date with ", pkgs)
end

function check(root=ROOT)
    manifests = tracked_manifests(root)
    problems = String[]
    for rel in manifests
        check_manifest(root, rel, problems)
    end
    isempty(manifests) && println("pinned-envs: no tracked Manifest.toml")
    if !isempty(problems)
        envs = join(unique(relpath(dirname(joinpath(root, p)), root) for p in manifests), " ")
        msg = join(["A committed Manifest is out of date with the in-repo packages it develops:";
                    "  - " .* problems;
                    "Re-pin and commit it in the same change (dev-reference §14):";
                    "  ~/github/nsiccha/Claude/pin-deps.sh $envs";
                    "  git add $(join(manifests, ' '))"], "\n")
        error(msg)
    end
    nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    check()
end
