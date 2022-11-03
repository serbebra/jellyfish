(function() {var implementors = {
"jf_plonk":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;PCSError&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.65.0/std/io/error/struct.Error.html\" title=\"struct std::io::error::Error\">Error</a>&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;SerializationError&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;RescueError&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_plonk/errors/enum.SnarkError.html\" title=\"enum jf_plonk::errors::SnarkError\">SnarkError</a>&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;CircuitError&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>&gt; for CircuitError"],["impl&lt;E, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.Proof.html\" title=\"struct jf_plonk::proof_system::structs::Proof\">Proof</a>&lt;E&gt;&gt; for <a class=\"struct\" href=\"https://doc.rust-lang.org/1.65.0/alloc/vec/struct.Vec.html\" title=\"struct alloc::vec::Vec\">Vec</a>&lt;E::Fq&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;E: PairingEngine&lt;G1Affine = GroupAffine&lt;P&gt;&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: SWModelParameters&lt;BaseField = E::Fq, ScalarField = E::Fr&gt;,</span>"],["impl&lt;E:&nbsp;PairingEngine&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.Proof.html\" title=\"struct jf_plonk::proof_system::structs::Proof\">Proof</a>&lt;E&gt;&gt; for <a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.BatchProof.html\" title=\"struct jf_plonk::proof_system::structs::BatchProof\">BatchProof</a>&lt;E&gt;"],["impl&lt;F:&nbsp;Field&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.ProofEvaluations.html\" title=\"struct jf_plonk::proof_system::structs::ProofEvaluations\">ProofEvaluations</a>&lt;F&gt;&gt; for <a class=\"struct\" href=\"https://doc.rust-lang.org/1.65.0/alloc/vec/struct.Vec.html\" title=\"struct alloc::vec::Vec\">Vec</a>&lt;F&gt;"],["impl&lt;E, F, P1, P2&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.VerifyingKey.html\" title=\"struct jf_plonk::proof_system::structs::VerifyingKey\">VerifyingKey</a>&lt;E&gt;&gt; for <a class=\"struct\" href=\"https://doc.rust-lang.org/1.65.0/alloc/vec/struct.Vec.html\" title=\"struct alloc::vec::Vec\">Vec</a>&lt;E::Fq&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;E: PairingEngine&lt;G1Affine = GroupAffine&lt;P1&gt;, G2Affine = GroupAffine&lt;P2&gt;, Fqe = Fp2&lt;F&gt;&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Fp2Parameters&lt;Fp = E::Fq&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;P1: SWModelParameters&lt;BaseField = E::Fq, ScalarField = E::Fr&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;P2: SWModelParameters&lt;BaseField = E::Fqe, ScalarField = E::Fr&gt;,</span>"]],
"jf_primitives":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.array.html\">[</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.u8.html\">u8</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.array.html\">; 32]</a>&gt; for <a class=\"struct\" href=\"jf_primitives/aead/struct.EncKey.html\" title=\"struct jf_primitives::aead::EncKey\">EncKey</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_primitives/aead/struct.EncKey.html\" title=\"struct jf_primitives::aead::EncKey\">EncKey</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.array.html\">[</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.u8.html\">u8</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.array.html\">; 32]</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.array.html\">[</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.usize.html\">usize</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.array.html\">; 4]</a>&gt; for <a class=\"struct\" href=\"jf_primitives/circuit/rescue/struct.RescueStateVar.html\" title=\"struct jf_primitives::circuit::rescue::RescueStateVar\">RescueStateVar</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_primitives/rescue/errors/enum.RescueError.html\" title=\"enum jf_primitives::rescue::errors::RescueError\">RescueError</a>&gt; for <a class=\"enum\" href=\"jf_primitives/errors/enum.PrimitivesError.html\" title=\"enum jf_primitives::errors::PrimitivesError\">PrimitivesError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;SerializationError&gt; for <a class=\"enum\" href=\"jf_primitives/errors/enum.PrimitivesError.html\" title=\"enum jf_primitives::errors::PrimitivesError\">PrimitivesError</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_primitives/merkle_tree/enum.NodePos.html\" title=\"enum jf_primitives::merkle_tree::NodePos\">NodePos</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.usize.html\">usize</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_primitives/merkle_tree/enum.NodePos.html\" title=\"enum jf_primitives::merkle_tree::NodePos\">NodePos</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.u8.html\">u8</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.u8.html\">u8</a>&gt; for <a class=\"enum\" href=\"jf_primitives/merkle_tree/enum.NodePos.html\" title=\"enum jf_primitives::merkle_tree::NodePos\">NodePos</a>"],["impl&lt;F:&nbsp;PrimeField&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.u64.html\">u64</a>&gt; for <a class=\"struct\" href=\"jf_primitives/merkle_tree/struct.NodeValue.html\" title=\"struct jf_primitives::merkle_tree::NodeValue\">NodeValue</a>&lt;F&gt;"],["impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_primitives/merkle_tree/enum.LookupResult.html\" title=\"enum jf_primitives::merkle_tree::LookupResult\">LookupResult</a>&lt;F, P&gt;&gt; for <a class=\"enum\" href=\"https://doc.rust-lang.org/1.65.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"enum\" href=\"https://doc.rust-lang.org/1.65.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.tuple.html\">(F, P)</a>&gt;&gt;"],["impl&lt;F:&nbsp;<a class=\"trait\" href=\"jf_primitives/rescue/trait.RescueParameter.html\" title=\"trait jf_primitives::rescue::RescueParameter\">RescueParameter</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_primitives/merkle_tree/struct.FilledMTBuilder.html\" title=\"struct jf_primitives::merkle_tree::FilledMTBuilder\">FilledMTBuilder</a>&lt;F&gt;&gt; for <a class=\"struct\" href=\"jf_primitives/merkle_tree/struct.MerkleTree.html\" title=\"struct jf_primitives::merkle_tree::MerkleTree\">MerkleTree</a>&lt;F&gt;"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;SerializationError&gt; for <a class=\"enum\" href=\"jf_primitives/pcs/errors/enum.PCSError.html\" title=\"enum jf_primitives::pcs::errors::PCSError\">PCSError</a>"],["impl&lt;F:&nbsp;PrimeField&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;F&gt; for <a class=\"struct\" href=\"jf_primitives/prf/struct.PrfKey.html\" title=\"struct jf_primitives::prf::PrfKey\">PrfKey</a>&lt;F&gt;"],["impl&lt;F:&nbsp;<a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/marker/trait.Copy.html\" title=\"trait core::marker::Copy\">Copy</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.slice.html\">[F]</a>&gt; for <a class=\"struct\" href=\"jf_primitives/rescue/struct.RescueVector.html\" title=\"struct jf_primitives::rescue::RescueVector\">RescueVector</a>&lt;F&gt;"],["impl&lt;F:&nbsp;<a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/marker/trait.Copy.html\" title=\"trait core::marker::Copy\">Copy</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.array.html\">[</a>F<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.array.html\">; 4]</a>&gt; for <a class=\"struct\" href=\"jf_primitives/rescue/struct.RescueVector.html\" title=\"struct jf_primitives::rescue::RescueVector\">RescueVector</a>&lt;F&gt;"],["impl&lt;F:&nbsp;PrimeField&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.array.html\">[</a><a class=\"struct\" href=\"jf_primitives/rescue/struct.RescueVector.html\" title=\"struct jf_primitives::rescue::RescueVector\">RescueVector</a>&lt;F&gt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.array.html\">; 4]</a>&gt; for <a class=\"struct\" href=\"jf_primitives/rescue/struct.RescueMatrix.html\" title=\"struct jf_primitives::rescue::RescueMatrix\">RescueMatrix</a>&lt;F&gt;"],["impl&lt;F:&nbsp;<a class=\"trait\" href=\"jf_primitives/rescue/trait.RescueParameter.html\" title=\"trait jf_primitives::rescue::RescueParameter\">RescueParameter</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_primitives/rescue/struct.PRP.html\" title=\"struct jf_primitives::rescue::PRP\">PRP</a>&lt;F&gt;&gt; for <a class=\"struct\" href=\"jf_primitives/rescue/struct.Permutation.html\" title=\"struct jf_primitives::rescue::Permutation\">Permutation</a>&lt;F&gt;"],["impl&lt;P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;GroupAffine&lt;P&gt;&gt; for <a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.VerKey.html\" title=\"struct jf_primitives::signatures::schnorr::VerKey\">VerKey</a>&lt;P&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters,</span>"],["impl&lt;P, F&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.SignKey.html\" title=\"struct jf_primitives::signatures::schnorr::SignKey\">SignKey</a>&lt;F&gt;&gt; for <a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.VerKey.html\" title=\"struct jf_primitives::signatures::schnorr::VerKey\">VerKey</a>&lt;P&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;ScalarField = F&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,</span>"],["impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.VerKey.html\" title=\"struct jf_primitives::signatures::schnorr::VerKey\">VerKey</a>&lt;P&gt;&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.tuple.html\">(F, F)</a><span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;BaseField = F&gt;,</span>"],["impl&lt;P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"struct\" href=\"jf_primitives/elgamal/struct.EncKey.html\" title=\"struct jf_primitives::elgamal::EncKey\">EncKey</a>&lt;P&gt;&gt; for (P::BaseField, P::BaseField)<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters,</span>"]],
"jf_relation":[["impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;GroupAffine&lt;P&gt;&gt; for <a class=\"struct\" href=\"jf_relation/gadgets/ecc/struct.Point.html\" title=\"struct jf_relation::gadgets::ecc::Point\">Point</a>&lt;F&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField + <a class=\"trait\" href=\"jf_relation/gadgets/ecc/trait.SWToTEConParam.html\" title=\"trait jf_relation::gadgets::ecc::SWToTEConParam\">SWToTEConParam</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: SWParam&lt;BaseField = F&gt;,</span>"],["impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;GroupAffine&lt;P&gt;&gt; for <a class=\"struct\" href=\"jf_relation/gadgets/ecc/struct.Point.html\" title=\"struct jf_relation::gadgets::ecc::Point\">Point</a>&lt;F&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;BaseField = F&gt;,</span>"],["impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;GroupAffine&lt;P&gt;&gt; for <a class=\"struct\" href=\"jf_relation/gadgets/ecc/struct.Point.html\" title=\"struct jf_relation::gadgets::ecc::Point\">Point</a>&lt;F&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: SWModelParameters&lt;BaseField = F&gt;,</span>"],["impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;GroupProjective&lt;P&gt;&gt; for <a class=\"struct\" href=\"jf_relation/gadgets/ecc/struct.Point.html\" title=\"struct jf_relation::gadgets::ecc::Point\">Point</a>&lt;F&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;BaseField = F&gt;,</span>"],["impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_relation/gadgets/ecc/struct.Point.html\" title=\"struct jf_relation::gadgets::ecc::Point\">Point</a>&lt;F&gt;&gt; for GroupAffine&lt;P&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;BaseField = F&gt;,</span>"],["impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_relation/gadgets/ecc/struct.Point.html\" title=\"struct jf_relation::gadgets::ecc::Point\">Point</a>&lt;F&gt;&gt; for GroupProjective&lt;P&gt;<span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;BaseField = F&gt;,</span>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_relation/constraint_system/struct.BoolVar.html\" title=\"struct jf_relation::constraint_system::BoolVar\">BoolVar</a>&gt; for <a class=\"type\" href=\"jf_relation/constraint_system/type.Variable.html\" title=\"type jf_relation::constraint_system::Variable\">Variable</a>"]],
"jf_utils":[["impl&lt;T:&nbsp;CanonicalSerialize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;T&gt; for <a class=\"struct\" href=\"jf_utils/struct.CanonicalBytes.html\" title=\"struct jf_utils::CanonicalBytes\">CanonicalBytes</a>"],["impl&lt;T:&nbsp;<a class=\"trait\" href=\"jf_utils/trait.Tagged.html\" title=\"trait jf_utils::Tagged\">Tagged</a> + CanonicalSerialize + CanonicalDeserialize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;T&gt; for <a class=\"struct\" href=\"jf_utils/struct.TaggedBlob.html\" title=\"struct jf_utils::TaggedBlob\">TaggedBlob</a>&lt;T&gt;"],["impl&lt;T:&nbsp;<a class=\"trait\" href=\"jf_utils/trait.Tagged.html\" title=\"trait jf_utils::Tagged\">Tagged</a> + CanonicalSerialize + CanonicalDeserialize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.65.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.65.0/std/primitive.reference.html\">&amp;</a>T&gt; for <a class=\"struct\" href=\"jf_utils/struct.TaggedBlob.html\" title=\"struct jf_utils::TaggedBlob\">TaggedBlob</a>&lt;T&gt;"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()