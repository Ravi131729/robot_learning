"""Command-line training entry point for the robot policy."""

import argparse
import pickle
import time
from pathlib import Path

import jax
import numpy as np
import torch

from configs.model_presets import PRESETS, get_preset
from data.abc_clip import load_abc_clip_text_embedder
from data.abc_dataset import ABCDataset, collate_abc_samples
from data.hf_dino import HFDinoV3Encoder, encode_preprocessed_camera_images
from data.image_preprocessing import preprocess_camera_dict
from data.model_batch import encode_abc_batch
from data.abc_official_batch import encode_official_batch
from data.abc_official_loader import create_abc_loader
from data.task_text import resolve_prompt
from data.types import PolicyBatch
from robot_policy.objectives.flow_matching import flow_matching_loss
from robot_policy.sampling.flow_sampler import sample_actions
from robot_policy.policy import init_policy_params
from training import create_optimizer, create_train_state, make_batch_train_step


def tree_nbytes(tree):
    """Return the host-side byte size of an arbitrary JAX pytree."""
    return sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(tree))


def torch_module_nbytes(module):
    """Return parameter and buffer bytes for a PyTorch module."""
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in module.buffers()
    )
    return parameter_bytes + buffer_bytes


def torch_module_parameter_counts(module):
    """Return total and trainable parameter counts for a PyTorch module."""
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def gibibytes(value):
    return value / (1024 ** 3)


def print_memory(label, dino=None, clip=None, params=None):
    """Print process, PyTorch CUDA, and JAX memory snapshots."""
    import os

    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss
        rss_text = f"RSS={gibibytes(rss):.2f} GiB"
    except ImportError:
        rss_text = "RSS=unavailable (install psutil)"

    torch_text = "Torch CUDA=unavailable"
    if torch.cuda.is_available():
        torch_text = (
            f"Torch allocated={gibibytes(torch.cuda.memory_allocated()):.2f} GiB, "
            f"reserved={gibibytes(torch.cuda.memory_reserved()):.2f} GiB, "
            f"peak={gibibytes(torch.cuda.max_memory_allocated()):.2f} GiB"
        )

    jax_text = "JAX device memory=unavailable"
    for device in jax.devices():
        if device.platform != "cpu":
            try:
                stats = device.memory_stats() or {}
                current = stats.get("bytes_in_use", stats.get("bytes_used"))
                limit = stats.get("bytes_limit")
                if current is not None:
                    jax_text = f"JAX allocated={gibibytes(current):.2f} GiB"
                if limit is not None:
                    jax_text += f", limit={gibibytes(limit):.2f} GiB"
            except Exception:
                pass
            break

    components = []
    if dino is not None:
        total, _ = torch_module_parameter_counts(dino.model)
        components.append(
            f"DINO params={gibibytes(torch_module_nbytes(dino.model)):.2f} GiB "
            f"({total / 1e6:.1f}M total, 0M active in policy training)"
        )
    if clip is not None and hasattr(clip, "encoder") and hasattr(clip.encoder, "model"):
        total, _ = torch_module_parameter_counts(clip.encoder.model)
        components.append(
            f"CLIP params={gibibytes(torch_module_nbytes(clip.encoder.model)):.2f} GiB "
            f"({total / 1e6:.1f}M total, 0M active in policy training)"
        )
    if params is not None:
        policy_bytes = tree_nbytes(params)
        policy_count = policy_bytes // 4
        components.append(
            f"active policy params={gibibytes(policy_bytes):.2f} GiB "
            f"({policy_count / 1e6:.1f}M trainable, FP32 estimate)"
        )

    detail = ", ".join(components)
    print(f"[memory:{label}] {rss_text}; {torch_text}; {jax_text}")
    if detail:
        print(f"  component sizes: {detail}")


def collect_memory_metrics(dino=None, clip=None, params=None):
    """Collect numeric memory metrics suitable for W&B."""
    import os

    metrics = {}
    try:
        import psutil
        metrics["memory/cpu_rss_gib"] = gibibytes(
            psutil.Process(os.getpid()).memory_info().rss
        )
    except ImportError:
        pass

    if torch.cuda.is_available():
        metrics.update({
            "memory/torch_allocated_gib": gibibytes(torch.cuda.memory_allocated()),
            "memory/torch_reserved_gib": gibibytes(torch.cuda.memory_reserved()),
            "memory/torch_peak_allocated_gib": gibibytes(
                torch.cuda.max_memory_allocated()
            ),
        })

    for device in jax.devices():
        if device.platform != "cpu":
            try:
                stats = device.memory_stats() or {}
                current = stats.get("bytes_in_use", stats.get("bytes_used"))
                limit = stats.get("bytes_limit")
                if current is not None:
                    metrics["memory/jax_allocated_gib"] = gibibytes(current)
                if limit is not None:
                    metrics["memory/jax_limit_gib"] = gibibytes(limit)
            except Exception:
                pass
            break

    if dino is not None:
        metrics["params/dino_millions"] = sum(
            parameter.numel() for parameter in dino.model.parameters()
        ) / 1e6
    if clip is not None and hasattr(clip, "encoder") and hasattr(clip.encoder, "model"):
        metrics["params/clip_millions"] = sum(
            parameter.numel() for parameter in clip.encoder.model.parameters()
        ) / 1e6
    if params is not None:
        metrics["params/policy_millions"] = tree_nbytes(params) / 4e6
    return metrics


def init_wandb(args, preset):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requires a working wandb installation; "
            "run `python -m pip install -U wandb protobuf<6`."
        ) from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            "preset_name": preset.name,
            "dit_depth": preset.dit_depth,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/ravi/robot_learning/cache")
    parser.add_argument("--split", default="train_sim")
    parser.add_argument(
        "--dino-model",
        default="/home/ravi/robot_learning/cache/dinov3-vitb16-pretrain-lvd1689m",
    )
    parser.add_argument(
        "--abc-root",
        default=None,
        help="local ABC checkout; defaults to ABC_REPO_ROOT or installed package",
    )
    parser.add_argument(
        "--clip-cache",
        default="/home/ravi/robot_learning/cache/clip",
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="debug")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--val-split", default="val_sim")
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="run validation every N training steps; 0 disables periodic evaluation",
    )
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--timing",
        action="store_true",
        help="report JAX compilation and per-update timings",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="report CPU/GPU memory snapshots during setup and training",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--wandb", action="store_true", help="enable W&B logging")
    parser.add_argument("--wandb-project", default="robot-learning-policy")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def save_checkpoint(path, train_state, args, losses):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(train_state.step),
        "params": jax.tree_util.tree_map(np.asarray, train_state.params),
        "opt_state": jax.tree_util.tree_map(np.asarray, train_state.opt_state),
        "losses": list(losses),
        "args": vars(args),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def sample_batch(dataset, rng, batch_size):
    indices = rng.integers(0, len(dataset), size=batch_size)
    return collate_abc_samples([dataset[int(index)] for index in indices])


def encode_abc_batch_with_timings(raw_batch, dino, clip):
    """Encode a batch while measuring each CPU/GPU input-pipeline stage."""
    timings = {}

    start = time.perf_counter()
    processed_images = preprocess_camera_dict(raw_batch.images)
    timings["image preprocessing"] = time.perf_counter() - start

    start = time.perf_counter()
    dino_tokens = encode_preprocessed_camera_images(dino, processed_images)
    timings["DINOv3 encoding"] = time.perf_counter() - start

    prompts = [
        resolve_prompt(task_name, instruction)
        for task_name, instruction in zip(
            raw_batch.task_names,
            raw_batch.instructions,
        )
    ]
    start = time.perf_counter()
    task_embeddings = clip.encode_prompts(prompts)
    timings["CLIP encoding"] = time.perf_counter() - start

    model_batch = PolicyBatch(
        dino_tokens=jax.numpy.asarray(dino_tokens),
        state=jax.numpy.asarray(raw_batch.state),
        task=jax.numpy.asarray(task_embeddings),
        actions=jax.numpy.asarray(raw_batch.actions),
    ).validate()

    start = time.perf_counter()
    model_batch = PolicyBatch(
        dino_tokens=jax.device_put(model_batch.dino_tokens),
        state=jax.device_put(model_batch.state),
        task=jax.device_put(model_batch.task),
        actions=jax.device_put(model_batch.actions),
    )
    for value in (
        model_batch.dino_tokens,
        model_batch.state,
        model_batch.task,
        model_batch.actions,
    ):
        value.block_until_ready()
    timings["CPU-to-GPU batch preparation"] = time.perf_counter() - start
    return model_batch, timings


@jax.jit
def eval_step(params, dino_tokens, state, task, actions, key):
    return flow_matching_loss(
        params,
        dino_tokens,
        state,
        task,
        actions,
        key,
    )


def evaluate_loader(params, loader, dino, clip, device, key, num_steps):
    """Evaluate flow loss on a fixed number of validation batches."""
    iterator = iter(loader)
    losses = []
    last_batch = None
    for step in range(num_steps):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        model_batch, _ = encode_official_batch(
            raw_batch,
            dino,
            clip,
            device=device,
        )
        last_batch = model_batch
        loss = eval_step(
            params,
            model_batch.dino_tokens,
            model_batch.state,
            model_batch.task,
            model_batch.actions,
            jax.random.fold_in(key, step),
        )
        loss.block_until_ready()
        losses.append(float(loss))
    return float(np.mean(losses)), last_batch


def main():
    args = parse_args()
    if args.batch_size < 1 or args.steps < 1:
        raise ValueError("--batch-size and --steps must be at least 1")
    if args.val_steps < 1 or args.sample_steps < 1:
        raise ValueError("--val-steps and --sample-steps must be at least 1")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be non-negative")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")

    preset = get_preset(args.preset)
    wandb_run = init_wandb(args, preset)
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"JAX devices: {jax.devices()}")
    print(f"Torch device: {torch_device}")
    print(f"Loading ABC official loader for {args.split} from {args.data_root}")
    loader = create_abc_loader(
        args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        abc_root=args.abc_root,
        seed=args.seed,
    )
    print(
        f"ABC loader batches: {len(loader)}, "
        f"workers: {args.num_workers}, pinned memory: true"
    )
    if args.memory:
        print_memory("dataset", params=None)

    dino = HFDinoV3Encoder.from_pretrained(
        args.dino_model,
        device=torch_device,
    )
    if args.memory:
        print_memory("after DINO", dino=dino)
    clip = load_abc_clip_text_embedder(
        abc_root=args.abc_root,
        cache_dir=args.clip_cache,
        device=torch_device,
    )
    if args.memory:
        print_memory("after CLIP", dino=dino, clip=clip)

    key = jax.random.PRNGKey(args.seed)
    model_key, train_key = jax.random.split(key)
    params = init_policy_params(model_key, dit_depth=preset.dit_depth)
    if args.memory:
        print_memory("after policy init", dino=dino, clip=clip, params=params)
    optimizer = create_optimizer(args.learning_rate, args.weight_decay)
    train_state = create_train_state(params, optimizer)
    train_step = make_batch_train_step(optimizer)
    loader_iterator = iter(loader)
    losses = []
    update_times = []
    val_loader = None
    best_val_loss = float("inf")
    measure_timings = args.timing or wandb_run is not None

    def get_val_loader():
        return create_abc_loader(
            args.data_root,
            split=args.val_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            abc_root=args.abc_root,
            seed=args.seed,
            train=False,
        )

    print(
        f"Training preset={preset.name}, depth={preset.dit_depth}, "
        f"batch_size={args.batch_size}, steps={args.steps}"
    )
    for step in range(args.steps):
        load_start = time.perf_counter()
        try:
            raw_batch = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            raw_batch = next(loader_iterator)
        load_time = time.perf_counter() - load_start
        if measure_timings:
            model_batch, pipeline_timings = encode_official_batch(
                raw_batch,
                dino,
                clip,
                device=jax.devices()[0],
                with_timings=True,
            )
        else:
            model_batch, _ = encode_official_batch(
                raw_batch,
                dino,
                clip,
                device=jax.devices()[0],
            )
        step_key = jax.random.fold_in(train_key, step)

        update_start = time.perf_counter()
        train_state, loss = train_step(train_state, model_batch, step_key)
        # Waiting here makes the timing include asynchronous device execution.
        loss.block_until_ready()
        update_time = time.perf_counter() - update_start
        update_times.append(update_time)
        loss_value = float(loss)
        losses.append(loss_value)
        if wandb_run is not None:
            metrics = {
                "train/loss": loss_value,
                "train/step_time_sec": update_time,
                "train/updates_per_sec": 1.0 / update_time,
                "train/samples_per_sec": args.batch_size / update_time,
                "train/learning_rate": args.learning_rate,
            }
            if measure_timings:
                metrics["timing/dataset_video_sec"] = load_time
                for stage, duration in pipeline_timings.items():
                    key = stage.lower().replace(" ", "_").replace("-", "")
                    metrics[f"timing/{key}_sec"] = duration
            if args.memory or wandb_run is not None:
                metrics.update(
                    collect_memory_metrics(dino, clip, train_state.params)
                )
            wandb_run.log(metrics, step=step + 1)
        if args.memory and step == 0:
            print_memory("after first update", dino=dino, clip=clip, params=train_state.params)

        if (step + 1) % args.log_every == 0 or step == 0:
            print(f"step={step + 1}/{args.steps} loss={loss_value:.6f}")
        if args.timing:
            print(f"  Dataset/video loading: {load_time:.4f}s")
            for stage, duration in pipeline_timings.items():
                print(f"  {stage}: {duration:.4f}s")
            if step == 0:
                print(
                    "  JAX compile + first update: "
                    f"{update_time:.4f}s"
                )
            else:
                print(f"  JAX update: {update_time:.4f}s")

        should_evaluate = (
            args.eval_every > 0
            and ((step + 1) % args.eval_every == 0 or step + 1 == args.steps)
        )
        if should_evaluate:
            if val_loader is None:
                val_loader = get_val_loader()
            val_loss, _ = evaluate_loader(
                train_state.params,
                val_loader,
                dino,
                clip,
                jax.devices()[0],
                jax.random.fold_in(train_key, 100000 + step),
                args.val_steps,
            )
            print(f"Validation step={step + 1}: flow-matching loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = Path(args.checkpoint_dir) / f"{preset.name}_best.pkl"
                save_checkpoint(best_path, train_state, args, losses)
                print(f"Saved best checkpoint: {best_path}")
            if wandb_run is not None:
                wandb_run.log(
                    {"validation/flow_matching_loss": val_loss},
                    step=step + 1,
                )

    checkpoint_path = Path(args.checkpoint_dir) / f"{preset.name}_latest.pkl"
    save_checkpoint(checkpoint_path, train_state, args, losses)
    print(f"Saved checkpoint: {checkpoint_path}")
    if args.timing and len(update_times) > 1:
        print(
            "Average steady-state JAX update: "
            f"{np.mean(update_times[1:]):.4f}s"
        )

    print(f"Running final validation on {args.val_split} ({args.val_steps} batches)")
    if val_loader is None:
        val_loader = get_val_loader()
    final_val_loss, sample_batch_for_inference = evaluate_loader(
        train_state.params,
        val_loader,
        dino,
        clip,
        jax.devices()[0],
        jax.random.fold_in(train_key, 300000),
        args.val_steps,
    )
    print(f"Final validation flow-matching loss: {final_val_loss:.6f}")
    if wandb_run is not None:
        wandb_run.log(
            {"validation/final_flow_matching_loss": final_val_loss},
            step=args.steps,
        )

    sampled_actions = sample_actions(
        sample_batch_for_inference.dino_tokens,
        sample_batch_for_inference.state,
        sample_batch_for_inference.task,
        train_state.params,
        jax.random.fold_in(train_key, 200000),
        num_steps=args.sample_steps,
    )
    sampled_actions.block_until_ready()
    assert sampled_actions.shape[1:] == (30, 14)
    assert np.isfinite(np.asarray(sampled_actions)).all()
    print(
        "Sampled actions: "
        f"shape={tuple(sampled_actions.shape)}, "
        f"range=[{float(sampled_actions.min()):.3f}, "
        f"{float(sampled_actions.max()):.3f}]"
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "inference/action_min": float(sampled_actions.min()),
                "inference/action_max": float(sampled_actions.max()),
                "inference/action_mean": float(sampled_actions.mean()),
            },
            step=args.steps,
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
