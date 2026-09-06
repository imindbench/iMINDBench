#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_experiments.sh FAMILY DATASET [MODEL|all] [grid options]
# Each model previews its task/target grid; --execute runs it, --count counts it.
# --limit applies per model. Any failed model stops the script.
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 FAMILY DATASET [MODEL|all] --config-dir DIR --output-root DIR [--execute]" >&2
  exit 2
fi
family=$1
dataset=$2
shift 2
selected_model=all
if [[ $# -gt 0 && $1 != --* ]]; then
  selected_model=$1
  shift
fi
case "$dataset" in
  neuroprobev2) rate=2048; subset=lite ;;
  kelesbyd2024) rate=1000; subset=full ;;
  berezutskayapippi2022) rate=2048; subset=high-cov ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
# Identity belongs to the selection above; tuning and task/target filters go below.
for arg in "$@"; do
  case "$arg" in
    --model|--model=*|--dataset|--dataset=*|--experiment|--experiment=*|--preprocessor|--preprocessor=*)
      echo "Choose FAMILY DATASET MODEL positionally; edit a script copy to change preprocessing." >&2
      exit 2 ;;
  esac
done
models=()
options=()
preprocessor="laplacian_multi_stft_${rate}Hz"
experiment=baseline
case "$family" in
  baselines)
    case "$dataset" in
      neuroprobev2) models=(logistic mlp cnn popt) ;;
      kelesbyd2024) models=(htnet_1000Hz popt) ;;
      berezutskayapippi2022) models=(popt htnet_2048Hz) ;;
    esac ;;
  brainbert)
    models=(linear_baseline)
    experiment=within_session
    preprocessor="laplacian_stft_brainbert_${rate}Hz"
    options=(--population brainbert --set '++preprocessor.chain.4.device=${model.device}') ;;
  barista)
    models=(barista)
    experiment=barista
    if [[ $rate == 1000 ]]; then
      preprocessor=laplacian_wav_session_HPF_global_robust_scalar_1000Hz_2048Hz_zscore
      options=(--set model.tokenizer.samp_frequency=2048 --set dataset.brain_area_key=label_destrieux)
    else
      preprocessor=laplacian_wav_session_HPF_global_robust_scalar_2048Hz_zscore
      if [[ $dataset == neuroprobev2 ]]; then
        options=(--set dataset.brain_area_key=localization_Destrieux)
      else
        options=(--set dataset.brain_area_key=label_destrieux)
      fi
    fi
    options+=(--population barista --task onset word_index volume) ;;
  hold_in|multisource)
    models=(popt)
    experiment=decodable
    options=(--population scaling --decodable-rule stft_or_htnet_500hz_val_mean0p60)
    if [[ $family == hold_in ]]; then
      options+=(--regime hold-in-session)
    fi ;;
  sample_efficiency)
    if [[ $dataset != neuroprobev2 ]]; then
      echo "sample_efficiency supports neuroprobev2 only" >&2; exit 2
    fi
    models=(logistic mlp cnn popt)
    options=(--sweep dataset.train_sample_fraction=1.0,0.5,0.25,0.125,0.0625) ;;
  stft_sweep)
    models=(logistic)
    experiment=within_session
    preprocessor="laplacian_stft_${rate}Hz"
    windows=256,512,1024
    if [[ $rate == 1000 ]]; then windows=125,250,500; fi
    options=(--sweep "preprocessor.chain.2.nperseg=$windows"
      --sweep preprocessor.chain.2.poverlap=0.50,0.75,0.875
      --sweep preprocessor.chain.2.max_frequency=100,150,250,400) ;;
  paper_multistft) models=(logistic mlp cnn popt) ;;
  paper_brainbert_stft)
    models=(logistic mlp cnn)
    preprocessor="laplacian_stft_classifier_${rate}Hz" ;;
  paper_htnet500)
    models=(htnet_500Hz)
    experiment=paper_htnet500
    preprocessor="laplacian_wav_HPF_global_robust_scalar_long_context_15s_${rate}Hzto500Hz" ;;
  paper_diver)
    models=(diver)
    experiment=paper_diver
    preprocessor="laplacian_wav_diverstyle_HPF_noSTD_long_context_15s_${rate}Hzto500Hz"
    if [[ $dataset == berezutskayapippi2022 ]]; then subset=full; fi ;;
  *) echo "Unknown experiment family: $family (see docs/EXPERIMENTS.md)" >&2; exit 2 ;;
esac
if [[ $selected_model != all ]]; then
  found=false
  for model in "${models[@]}"; do
    if [[ $model == "$selected_model" ]]; then found=true; fi
  done
  if [[ $found == false ]]; then
    echo "Model must be one of: ${models[*]}" >&2; exit 2
  fi
  models=("$selected_model")
fi
dataset_config=$dataset
if [[ $family == multisource ]]; then dataset_config="${dataset}_multisource_train"; fi
for model in "${models[@]}"; do
  model_options=()
  model_preprocessor=$preprocessor
  model_experiment=$experiment
  case "$family" in
    paper_multistft|paper_brainbert_stft) model_experiment="$family/$model" ;;
  esac
  if [[ $family == baselines && $model == htnet_* ]]; then
    model_preprocessor="laplacian_wav_${rate}Hz"
  fi
  if [[ $family == paper_multistft && $model == popt && $dataset != neuroprobev2 ]]; then
    model_options=(--set dataset.max_train_samples_per_subject=auto
      --set dataset.train_decodable_subject_sessions_only=false)
  fi
  python -m imindbench.launch \
    --dataset "$dataset_config" --model "$model" --preprocessor "$model_preprocessor" \
    --experiment "$model_experiment" --subset "$subset" --paths local \
    --output-group "${family}_${dataset}" \
    "${options[@]}" "${model_options[@]}" "$@"
done
