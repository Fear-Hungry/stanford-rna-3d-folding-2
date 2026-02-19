from __future__ import annotations

import json
import sys
from pathlib import Path

from .assets_fetch import fetch_pretrained_assets
from .boltz1_offline import predict_boltz1_offline
from .chai1_offline import predict_chai1_offline
from .chemical_features import prepare_chemical_features
from .cli_parser import build_parser
from .description_family import infer_description_family
from .embedding_index import build_embedding_index
from .ensemble.qa_ranker_se3 import rank_se3_ensemble
from .ensemble.select_top5 import select_top5_se3
from .evaluation import score_local_bestof5
from .evaluation.kaggle_oracle import score_local_kaggle_official
from .experiments import run_experiment
from .errors import PipelineError
from .homology_eval import evaluate_homology_folds
from .homology_folds import build_homology_folds
from .hybrid_router import build_hybrid_candidates
from .hybrid_select import select_top5_hybrid
from .minimization import minimize_ensemble
from .pairings import derive_pairings_from_chemical
from .phase2_assets import build_phase2_assets_manifest
from .phase2_configs import write_phase2_model_configs
from .rnapro_offline import predict_rnapro_offline
from .rnapro_support import prepare_rnapro_support_files
from .reranker import score_template_reranker, train_template_reranker
from .retrieval_latent import retrieve_templates_latent
from .submission import check_submission, export_submission
from .submit_kaggle_notebook import submit_kaggle_notebook
from .submit_readiness import evaluate_submit_readiness
from .se3_pipeline import sample_se3_ensemble, train_se3_generator
from .tbm import predict_tbm
from .template_db import build_template_db
from .training.data_lab import prepare_phase1_data_lab
from .wheelhouse import build_wheelhouse


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo = _repo_root()
    try:
        if args.command == "build-template-db":
            out = build_template_db(
                repo_root=repo,
                external_templates_path=(repo / args.external_templates).resolve(),
                out_dir=(repo / args.out_dir).resolve(),
            )
            _print_json(
                {
                    "templates": str(out.templates_path),
                    "template_index": str(out.template_index_path),
                    "manifest": str(out.manifest_path),
                }
            )
            return 0

        if args.command == "build-embedding-index":
            out = build_embedding_index(
                repo_root=repo,
                template_index_path=(repo / args.template_index).resolve(),
                out_dir=(repo / args.out_dir).resolve(),
                embedding_dim=int(args.embedding_dim),
                encoder=str(args.encoder),
                model_path=None if args.model_path is None else (repo / args.model_path).resolve(),
                ann_engine=str(args.ann_engine),
            )
            _print_json(
                {
                    "template_embeddings": str(out.embeddings_path),
                    "ann_index": None if out.index_path is None else str(out.index_path),
                    "manifest": str(out.manifest_path),
                }
            )
            return 0

        if args.command == "infer-description-family":
            out = infer_description_family(
                repo_root=repo,
                targets_path=(repo / args.targets).resolve(),
                out_dir=(repo / args.out_dir).resolve(),
                backend=str(args.backend),
                llm_model_path=None if args.llm_model_path is None else (repo / args.llm_model_path).resolve(),
                template_family_map_path=None if args.template_family_map is None else (repo / args.template_family_map).resolve(),
            )
            _print_json(
                {
                    "target_family": str(out.target_family_path),
                    "family_prior": None if out.family_prior_path is None else str(out.family_prior_path),
                    "manifest": str(out.manifest_path),
                }
            )
            return 0

        if args.command == "prepare-chemical-features":
            out = prepare_chemical_features(
                repo_root=repo,
                quickstart_path=(repo / args.quickstart).resolve(),
                out_path=(repo / args.out).resolve(),
            )
            _print_json({"features": str(out.features_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "derive-pairings-from-chemical":
            out = derive_pairings_from_chemical(
                repo_root=repo,
                chemical_features_path=(repo / args.chemical_features).resolve(),
                out_path=(repo / args.out).resolve(),
            )
            _print_json({"pairings": str(out.pairings_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "build-homology-folds":
            out = build_homology_folds(
                repo_root=repo,
                train_targets_path=(repo / args.train_targets).resolve(),
                pdb_sequences_path=(repo / args.pdb_sequences).resolve(),
                train_labels_path=None if args.train_labels is None else (repo / args.train_labels).resolve(),
                usalign_bin=str(args.usalign_bin),
                tm_threshold=float(args.tm_threshold),
                usalign_timeout_seconds=int(args.usalign_timeout_seconds),
                out_dir=(repo / args.out_dir).resolve(),
                backend=str(args.backend),
                identity_threshold=float(args.identity_threshold),
                coverage_threshold=float(args.coverage_threshold),
                n_folds=int(args.n_folds),
                chain_separator=str(args.chain_separator),
                mmseqs_bin=str(args.mmseqs_bin),
                cdhit_bin=str(args.cdhit_bin),
                domain_labels_path=None if args.domain_labels is None else (repo / args.domain_labels).resolve(),
                domain_column=str(args.domain_column),
                description_column=str(args.description_column),
                strict_domain_stratification=not bool(args.allow_no_domain_stratification),
            )
            _print_json({"clusters": str(out.clusters_path), "train_folds": str(out.train_folds_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "evaluate-homology-folds":
            out = evaluate_homology_folds(
                repo_root=repo,
                train_folds_path=(repo / args.train_folds).resolve(),
                target_metrics_path=(repo / args.target_metrics).resolve(),
                report_path=(repo / args.report).resolve(),
                orphan_labels_path=None if args.orphan_labels is None else (repo / args.orphan_labels).resolve(),
                retrieval_path=None if args.retrieval is None else (repo / args.retrieval).resolve(),
                metric_column=None if args.metric_column is None else str(args.metric_column),
                retrieval_score_column=None if args.retrieval_score_column is None else str(args.retrieval_score_column),
                orphan_score_threshold=float(args.orphan_score_threshold),
                orphan_weight=float(args.orphan_weight),
            )
            _print_json({"report": str(out.report_path)})
            return 0

        if args.command == "retrieve-templates-latent":
            out = retrieve_templates_latent(
                repo_root=repo,
                template_index_path=(repo / args.template_index).resolve(),
                template_embeddings_path=(repo / args.template_embeddings).resolve(),
                targets_path=(repo / args.targets).resolve(),
                out_path=(repo / args.out).resolve(),
                top_k=int(args.top_k),
                encoder=str(args.encoder),
                embedding_dim=int(args.embedding_dim),
                model_path=None if args.model_path is None else (repo / args.model_path).resolve(),
                ann_engine=str(args.ann_engine),
                faiss_index_path=None if args.faiss_index is None else (repo / args.faiss_index).resolve(),
                family_prior_path=None if args.family_prior is None else (repo / args.family_prior).resolve(),
                weight_embed=float(args.weight_embed),
                weight_llm=float(args.weight_llm),
                weight_seq=float(args.weight_seq),
            )
            _print_json({"candidates": str(out.candidates_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "train-template-reranker":
            out = train_template_reranker(
                repo_root=repo,
                candidates_path=(repo / args.candidates).resolve(),
                chemical_features_path=(repo / args.chemical_features).resolve(),
                out_dir=(repo / args.out_dir).resolve(),
                labels_path=None if args.labels is None else (repo / args.labels).resolve(),
                epochs=int(args.epochs),
                learning_rate=float(args.learning_rate),
                seed=int(args.seed),
            )
            _print_json({"model": str(out.model_path), "config": str(out.config_path), "metrics": str(out.metrics_path)})
            return 0

        if args.command == "score-template-reranker":
            out = score_template_reranker(
                repo_root=repo,
                candidates_path=(repo / args.candidates).resolve(),
                chemical_features_path=(repo / args.chemical_features).resolve(),
                model_path=(repo / args.model).resolve(),
                config_path=(repo / args.config).resolve(),
                out_path=(repo / args.out).resolve(),
                top_k=None if args.top_k is None else int(args.top_k),
            )
            _print_json({"scored": str(out.scored_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "predict-tbm":
            out = predict_tbm(
                repo_root=repo,
                retrieval_path=(repo / args.retrieval).resolve(),
                templates_path=(repo / args.templates).resolve(),
                targets_path=(repo / args.targets).resolve(),
                out_path=(repo / args.out).resolve(),
                n_models=int(args.n_models),
            )
            _print_json({"predictions": str(out.predictions_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "export-submission":
            out = export_submission(
                sample_path=(repo / args.sample).resolve(),
                predictions_long_path=(repo / args.predictions).resolve(),
                out_path=(repo / args.out).resolve(),
            )
            _print_json({"submission": str(out.submission_path)})
            return 0

        if args.command == "check-submission":
            check_submission(sample_path=(repo / args.sample).resolve(), submission_path=(repo / args.submission).resolve())
            _print_json({"ok": True, "submission": str((repo / args.submission).resolve())})
            return 0

        if args.command == "score-local-bestof5":
            out = score_local_bestof5(
                ground_truth_path=(repo / args.ground_truth).resolve(),
                submission_path=(repo / args.submission).resolve(),
                usalign_path=(repo / args.usalign_bin).resolve(),
                score_json_path=(repo / args.score_json).resolve(),
                report_path=(None if args.report is None else (repo / args.report).resolve()),
                timeout_seconds=int(args.timeout_seconds),
                ground_truth_mode=str(args.ground_truth_mode),
            )
            _print_json(
                {
                    "score": float(out.score),
                    "n_targets": int(out.n_targets),
                    "score_json": str(out.score_json_path),
                    "report": str(out.report_path),
                }
            )
            return 0

        if args.command == "score-local-kaggle-official":
            out = score_local_kaggle_official(
                ground_truth_path=(repo / args.ground_truth).resolve(),
                submission_path=(repo / args.submission).resolve(),
                score_json_path=(repo / args.score_json).resolve(),
                report_path=(repo / args.report).resolve(),
                metric_path=(None if args.metric_py is None else (repo / args.metric_py).resolve()),
            )
            _print_json(
                {
                    "score": float(out.score),
                    "score_json": str(out.score_json_path),
                    "report": str(out.report_path),
                }
            )
            return 0

        if args.command == "submit-kaggle-notebook":
            out = submit_kaggle_notebook(
                competition=str(args.competition),
                notebook_ref=str(args.notebook_ref),
                notebook_version=str(args.notebook_version),
                notebook_file=str(args.notebook_file),
                sample_path=(repo / args.sample).resolve(),
                submission_path=(repo / args.submission).resolve(),
                notebook_output_path=(repo / args.notebook_output_path).resolve(),
                score_json_path=(repo / args.score_json).resolve(),
                baseline_score=float(args.baseline_score),
                message=str(args.message),
                execute_submit=bool(args.execute_submit),
            )
            _print_json({"report": str(out.report_path)})
            return 0

        if args.command == "build-phase2-assets":
            out = build_phase2_assets_manifest(
                repo_root=repo,
                assets_dir=(repo / args.assets_dir).resolve(),
                manifest_path=None if args.manifest is None else (repo / args.manifest).resolve(),
            )
            _print_json({"manifest": str(out.manifest_path)})
            return 0

        if args.command == "fetch-pretrained-assets":
            out = fetch_pretrained_assets(
                repo_root=repo,
                assets_dir=(repo / args.assets_dir).resolve(),
                include=list(args.include),
                dry_run=bool(args.dry_run),
                timeout_seconds=int(args.timeout_seconds),
                max_bytes=None if args.max_bytes is None else int(args.max_bytes),
            )
            payload = dict(out.payload)
            payload["manifest"] = str(out.manifest_path)
            _print_json(payload)
            return 0

        if args.command == "prepare-rnapro-support-files":
            codes = [c.strip() for c in str(args.codes).split(",") if c.strip()]
            out = prepare_rnapro_support_files(
                repo_root=repo,
                model_dir=(repo / args.model_dir).resolve(),
                codes=codes,
                components_cif_gz_url=str(args.components_url),
                timeout_seconds=int(args.timeout_seconds),
                overwrite=bool(args.overwrite),
            )
            _print_json({"model_dir": str(out.model_dir), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "write-phase2-model-configs":
            out = write_phase2_model_configs(
                repo_root=repo,
                assets_dir=(repo / args.assets_dir).resolve(),
                chain_separator=str(args.chain_separator),
                manifest_path=None if args.manifest is None else (repo / args.manifest).resolve(),
            )
            _print_json({"manifest": str(out.manifest_path)})
            return 0

        if args.command == "build-wheelhouse":
            out = build_wheelhouse(
                repo_root=repo,
                wheels_dir=(repo / args.wheels_dir).resolve(),
                python_version=str(args.python_version),
                platform=str(args.platform),
                profile=str(args.profile),
                include_project_wheel=not bool(args.no_project_wheel),
                timeout_seconds=int(args.timeout_seconds),
                manifest_path=None if args.manifest is None else (repo / args.manifest).resolve(),
            )
            _print_json({"wheels_dir": str(out.wheels_dir), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "predict-rnapro-offline":
            out = predict_rnapro_offline(
                repo_root=repo,
                model_dir=(repo / args.model_dir).resolve(),
                targets_path=(repo / args.targets).resolve(),
                out_path=(repo / args.out).resolve(),
                n_models=int(args.n_models),
            )
            _print_json({"predictions": str(out.predictions_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "predict-chai1-offline":
            out = predict_chai1_offline(
                repo_root=repo,
                model_dir=(repo / args.model_dir).resolve(),
                targets_path=(repo / args.targets).resolve(),
                out_path=(repo / args.out).resolve(),
                n_models=int(args.n_models),
            )
            _print_json({"predictions": str(out.predictions_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "predict-boltz1-offline":
            out = predict_boltz1_offline(
                repo_root=repo,
                model_dir=(repo / args.model_dir).resolve(),
                targets_path=(repo / args.targets).resolve(),
                out_path=(repo / args.out).resolve(),
                n_models=int(args.n_models),
            )
            _print_json({"predictions": str(out.predictions_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "prepare-phase1-data-lab":
            out = prepare_phase1_data_lab(
                repo_root=repo,
                targets_path=(repo / args.targets).resolve(),
                pairings_path=(repo / args.pairings).resolve(),
                chemical_features_path=(repo / args.chemical_features).resolve(),
                labels_path=(repo / args.labels).resolve(),
                out_dir=(repo / args.out_dir).resolve(),
                thermo_backend=str(args.thermo_backend),
                rnafold_bin=str(args.rnafold_bin),
                linearfold_bin=str(args.linearfold_bin),
                msa_backend=str(args.msa_backend),
                mmseqs_bin=str(args.mmseqs_bin),
                mmseqs_db=str(args.mmseqs_db),
                chain_separator=str(args.chain_separator),
                max_msa_sequences=int(args.max_msa_sequences),
                max_cov_positions=int(args.max_cov_positions),
                max_cov_pairs=int(args.max_cov_pairs),
                num_workers=int(args.workers),
            )
            _print_json(
                {
                    "training_store": str(out.store_path),
                    "training_store_manifest": str(out.store_manifest_path),
                    "manifest": str(out.manifest_path),
                    "thermo_cache_dir": str(out.thermo_cache_dir),
                    "msa_cache_dir": str(out.msa_cache_dir),
                }
            )
            return 0

        if args.command == "train-se3-generator":
            out = train_se3_generator(
                repo_root=repo,
                targets_path=(repo / args.targets).resolve(),
                pairings_path=(repo / args.pairings).resolve(),
                chemical_features_path=(repo / args.chemical_features).resolve(),
                labels_path=(repo / args.labels).resolve(),
                config_path=(repo / args.config).resolve(),
                out_dir=(repo / args.out_dir).resolve(),
                seed=int(args.seed),
                training_store_path=(None if args.training_store is None else (repo / args.training_store).resolve()),
            )
            _print_json(
                {
                    "model_dir": str(out.model_dir),
                    "manifest": str(out.manifest_path),
                    "metrics": str(out.metrics_path),
                    "config_effective": str(out.config_effective_path),
                }
            )
            return 0

        if args.command == "sample-se3-ensemble":
            out = sample_se3_ensemble(
                repo_root=repo,
                model_dir=(repo / args.model_dir).resolve(),
                targets_path=(repo / args.targets).resolve(),
                pairings_path=(repo / args.pairings).resolve(),
                chemical_features_path=(repo / args.chemical_features).resolve(),
                out_path=(repo / args.out).resolve(),
                method=str(args.method),
                n_samples=int(args.n_samples),
                seed=int(args.seed),
            )
            _print_json({"candidates": str(out.candidates_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "rank-se3-ensemble":
            out = rank_se3_ensemble(
                repo_root=repo,
                candidates_path=(repo / args.candidates).resolve(),
                out_path=(repo / args.out).resolve(),
                qa_config_path=None if args.qa_config is None else (repo / args.qa_config).resolve(),
                chemical_features_path=(None if args.chemical_features is None else (repo / args.chemical_features).resolve()),
                diversity_lambda=float(args.diversity_lambda),
            )
            _print_json({"ranked": str(out.ranked_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "select-top5-se3":
            out = select_top5_se3(
                repo_root=repo,
                ranked_path=(repo / args.ranked).resolve(),
                out_path=(repo / args.out).resolve(),
                n_models=int(args.n_models),
                diversity_lambda=float(args.diversity_lambda),
            )
            _print_json({"predictions": str(out.predictions_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "build-hybrid-candidates":
            if args.se3 is not None and (args.se3_flash is None or args.se3_mamba is None):
                print(
                    "[HYBRID_ROUTER] [src/rna3d_local/cli.py:main] uso de --se3 legado detectado; prefira --se3-flash e --se3-mamba",
                    file=sys.stderr,
                )
            out = build_hybrid_candidates(
                repo_root=repo,
                targets_path=(repo / args.targets).resolve(),
                retrieval_path=(repo / args.retrieval).resolve(),
                tbm_path=(repo / args.tbm).resolve(),
                out_path=(repo / args.out).resolve(),
                routing_path=(repo / args.routing_out).resolve(),
                template_score_threshold=float(args.template_score_threshold),
                short_max_len=int(args.short_max_len),
                medium_max_len=int(args.medium_max_len),
                ultra_long_seq_threshold=(None if args.ultra_long_seq_threshold is None else int(args.ultra_long_seq_threshold)),
                rnapro_path=None if args.rnapro is None else (repo / args.rnapro).resolve(),
                chai1_path=None if args.chai1 is None else (repo / args.chai1).resolve(),
                boltz1_path=None if args.boltz1 is None else (repo / args.boltz1).resolve(),
                se3_path=None if args.se3 is None else (repo / args.se3).resolve(),
                se3_flash_path=None if args.se3_flash is None else (repo / args.se3_flash).resolve(),
                se3_mamba_path=None if args.se3_mamba is None else (repo / args.se3_mamba).resolve(),
            )
            _print_json(
                {
                    "candidates": str(out.candidates_path),
                    "routing": str(out.routing_path),
                    "manifest": str(out.manifest_path),
                }
            )
            return 0

        if args.command == "select-top5-hybrid":
            out = select_top5_hybrid(
                repo_root=repo,
                candidates_path=(repo / args.candidates).resolve(),
                out_path=(repo / args.out).resolve(),
                n_models=int(args.n_models),
                diversity_lambda=float(args.diversity_lambda),
            )
            _print_json({"predictions": str(out.predictions_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "minimize-ensemble":
            out = minimize_ensemble(
                repo_root=repo,
                predictions_path=(repo / args.predictions).resolve(),
                out_path=(repo / args.out).resolve(),
                backend=str(args.backend),
                max_iterations=int(args.max_iterations),
                bond_length_angstrom=float(args.bond_length_angstrom),
                bond_force_k=float(args.bond_force_k),
                angle_force_k=float(args.angle_force_k),
                angle_target_deg=float(args.angle_target_deg),
                vdw_min_distance_angstrom=float(args.vdw_min_distance_angstrom),
                vdw_epsilon=float(args.vdw_epsilon),
                position_restraint_k=float(args.position_restraint_k),
                openmm_platform=None if args.openmm_platform is None else str(args.openmm_platform),
            )
            _print_json({"predictions": str(out.predictions_path), "manifest": str(out.manifest_path)})
            return 0

        if args.command == "evaluate-submit-readiness":
            out = evaluate_submit_readiness(
                repo_root=repo,
                sample_path=(repo / args.sample).resolve(),
                submission_path=(repo / args.submission).resolve(),
                score_json_path=(repo / args.score_json).resolve(),
                baseline_score=float(args.baseline_score),
                report_path=(repo / args.report).resolve(),
                fail_on_disallow=not bool(args.allow_disallow),
            )
            _print_json({"allowed": bool(out.allowed), "report": str(out.report_path)})
            return 0

        if args.command == "run-experiment":
            out = run_experiment(
                repo_root=repo,
                recipe_path=(repo / args.recipe).resolve(),
                runs_dir=(repo / args.runs_dir).resolve(),
                tag_override=(None if args.tag is None else str(args.tag)),
                var_overrides=list(args.var),
                dry_run=bool(args.dry_run),
            )
            _print_json(
                {
                    "dry_run": bool(args.dry_run),
                    "run_dir": None if out is None else str(out.run_dir),
                    "recipe_resolved": None if out is None else str(out.recipe_resolved_path),
                    "meta": None if out is None else str(out.meta_path),
                    "report": None if out is None else str(out.report_path),
                }
            )
            return 0

        parser.print_help()
        return 2
    except PipelineError as exc:
        print(str(exc))
        return 1
