import argparse
from typing import Optional

import onnx
import onnxscript
import torch
from onnx.defs import OpSchema
from onnxscript import ir
from onnxscript import opset22 as opset22

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transformations.library import (
    match_causal_attn_mask,
    match_eager_attention,
    match_grouped_attention,
    match_moe_pattern,
    match_repeat_kv,
    match_rope_pattern,
)


def custom_simple_linear_op(input: ir.Tensor, weight: ir.Tensor, bias: Optional[ir.Tensor]):
    if bias is None:
        return opset22.MatMul(input, weight)
    return opset22.Add(opset22.MatMul(input, weight), bias)


custom_rope_schema = OpSchema(
    name="rope_with_explicit_cos_sin",
    domain="auto_deploy",
    since_version=1,
    doc="Rope with explicit cos and sin caches.",
    inputs=[
        OpSchema.FormalParameter(
            name="q",
            description="Q tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="k",
            description="K tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="cos",
            description="Cos cache",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="sin",
            description="Sin cache",
            type_str="T",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Output tensor",
            type_str="T",
        )
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="unsqueeze_dim",
            type=OpSchema.AttrType.INT,
            description="Unsqueeze dimension. Must be 1 or 2.",
            required=True,
        ),
    ],
)
onnx.defs.register_schema(custom_rope_schema)


def custom_rope_op(
    q: ir.Tensor, k: ir.Tensor, cos: ir.Tensor, sin: ir.Tensor, unsqueeze_dim: int = 1
):
    auto_deploy_op = onnxscript.values.Opset(domain="auto_deploy", version=1)
    return auto_deploy_op.rope_with_explicit_cos_sin(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


custom_grouped_sdpa_schema = OpSchema(
    name="torch_attention_grouped_sdpa",
    domain="auto_deploy",
    since_version=1,
    doc="Grouped SDPA attention.",
    inputs=[
        OpSchema.FormalParameter(
            name="query",
            description="Q tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="key",
            description="K tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="value",
            description="Value tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="attn_mask",
            description="Attention mask",
            type_str="T",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Output tensor",
            type_str="T",
        )
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="dropout_p",
            type=OpSchema.AttrType.FLOAT,
            description="Dropout probability.",
            required=True,
        ),
        OpSchema.Attribute(
            name="is_causal",
            type=OpSchema.AttrType.INT,
            description="Is causal.",
            required=True,
        ),
        OpSchema.Attribute(
            name="scale",
            type=OpSchema.AttrType.FLOAT,
            description="Scale",
            required=True,
        ),
    ],
)
onnx.defs.register_schema(custom_grouped_sdpa_schema)


def custom_grouped_sdpa_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    auto_deploy_op = onnxscript.values.Opset(domain="auto_deploy", version=1)
    return auto_deploy_op.torch_attention_grouped_sdpa(
        query,
        key,
        value,
        attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="The HF model to use for onnx export.",
    )
    parser.add_argument(
        "--output_onnx",
        type=str,
        default="output.onnx",
        help="The output onnx file name.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        help="The torch dtype to use for the model.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The max sequence length to use for the model.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="The max batch size to use for the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use for the model.",
    )
    args = parser.parse_args()

    print(f"Constructing model from {args.model}")
    factory = ModelFactoryRegistry.get("AutoModelForCausalLM")(
        model=args.model,
        model_kwargs={"torch_dtype": args.torch_dtype},
        tokenizer=args.model,
        tokenizer_kwargs={"torch_dtype": args.torch_dtype},
        skip_loading_weights=False,
        max_seq_len=args.max_seq_len,
    )
    model = factory.build_model(device=args.device)

    print("Exporting model to GraphModule")
    seq_info = SequenceInfo(
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        page_size=args.max_seq_len,
        max_num_tokens=None,
    )
    cm = CachedSequenceInterface(
        sequence_info=seq_info,
        device=args.device,
    )
    egm = torch_export_to_gm(model, args=cm.args)
    del model

    print("Applying transformations to GraphModule")

    # Match MoE pattern
    egm = match_moe_pattern(egm)

    # Match repeat_kv pattern
    egm = match_repeat_kv(egm)

    # Match eager attention pattern
    egm = match_eager_attention(egm)

    # Match grouped attention pattern
    egm = match_grouped_attention(egm)

    # Match and optimize causal attention masks
    egm = match_causal_attn_mask(egm)

    # Match rope
    egm, _ = match_rope_pattern(egm)

    # Export to ONNX
    print("Exporting to ONNX")
    torch.onnx.export(
        egm,
        args=cm.args,
        f=args.output_onnx,
        verbose=True,
        dynamo=True,
        custom_translation_table={
            torch.ops.auto_deploy.torch_linear_simple.default: custom_simple_linear_op,
            torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin.default: custom_rope_op,
            torch.ops.auto_deploy.torch_attention_grouped_sdpa.default: custom_grouped_sdpa_op,
        },
    )

    print(f"Exported ONNX to {args.output_onnx}")


if __name__ == "__main__":
    main()
