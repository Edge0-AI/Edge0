import Edge0Core
import MLX

/// Parallel, unmerged LoRA used by the official Edge0 deployment.
/// Base INT4 weights stay untouched; fp16 rank-16 adapters are added at run time.
struct ParallelLinear {
    let base: QuantizedExpertLinear
    private let loraA: MLXArray?
    private let loraB: MLXArray?
    private let scale: Float

    init(base: QuantizedExpertLinear, loraStore: ExpertTensorStore? = nil,
         prefix: String? = nil, rank: Int = 16, alpha: Float = 32) throws {
        self.base = base
        scale = alpha / Float(rank)
        if let loraStore, let prefix {
            let a = try loraStore.mlxArrayCopying(named: prefix + ".lora_A")
            let b = try loraStore.mlxArrayCopying(named: prefix + ".lora_B")
            guard a.shape == [rank, base.inputSize],
                  b.shape == [base.outputSize, rank] else {
                throw M1Error.invalid("LoRA dimensions disagree for \(prefix)")
            }
            loraA = a
            loraB = b
            eval(a, b)
        } else {
            loraA = nil
            loraB = nil
        }
    }

    var inputSize: Int { base.inputSize }
    var outputSize: Int { base.outputSize }

    func callAsFunction(_ input: MLXArray) throws -> MLXArray {
        let output = try base(input)
        guard let loraA, let loraB else { return output }
        let x = input.asType(loraA.dtype)
        let delta = matmul(matmul(x, loraA.T), loraB.T) * scale
        return output + delta.asType(output.dtype)
    }
}

struct ParallelExpert {
    let up: ParallelLinear
    let gate: ParallelLinear
    let down: ParallelLinear

    init(base: QuantizedExpert, loraStore: ExpertTensorStore?, prefix: String) throws {
        up = try ParallelLinear(base: base.up, loraStore: loraStore,
                                prefix: prefix + ".up_proj")
        gate = try ParallelLinear(base: base.gate, loraStore: loraStore,
                                  prefix: prefix + ".gate_proj")
        down = try ParallelLinear(base: base.down, loraStore: loraStore,
                                  prefix: prefix + ".down_proj")
    }

    func callAsFunction(_ input: MLXArray) throws -> MLXArray {
        let g = try gate(input)
        return try down((g * sigmoid(g)) * up(input))
    }
}
