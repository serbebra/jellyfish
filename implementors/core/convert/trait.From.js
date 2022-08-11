(function() {var implementors = {};
implementors["jf_plonk"] = [{"text":"impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;GroupAffine&lt;P&gt;&gt; for <a class=\"struct\" href=\"jf_plonk/circuit/customized/ecc/struct.Point.html\" title=\"struct jf_plonk::circuit::customized::ecc::Point\">Point</a>&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField + <a class=\"trait\" href=\"jf_plonk/circuit/customized/ecc/trait.SWToTEConParam.html\" title=\"trait jf_plonk::circuit::customized::ecc::SWToTEConParam\">SWToTEConParam</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: SWParam&lt;BaseField = F&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["jf_plonk::circuit::customized::ecc::Point"]},{"text":"impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;GroupAffine&lt;P&gt;&gt; for <a class=\"struct\" href=\"jf_plonk/circuit/customized/ecc/struct.Point.html\" title=\"struct jf_plonk::circuit::customized::ecc::Point\">Point</a>&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;BaseField = F&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["jf_plonk::circuit::customized::ecc::Point"]},{"text":"impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;GroupAffine&lt;P&gt;&gt; for <a class=\"struct\" href=\"jf_plonk/circuit/customized/ecc/struct.Point.html\" title=\"struct jf_plonk::circuit::customized::ecc::Point\">Point</a>&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: SWModelParameters&lt;BaseField = F&gt;,&nbsp;</span>","synthetic":false,"types":["jf_plonk::circuit::customized::ecc::Point"]},{"text":"impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;GroupProjective&lt;P&gt;&gt; for <a class=\"struct\" href=\"jf_plonk/circuit/customized/ecc/struct.Point.html\" title=\"struct jf_plonk::circuit::customized::ecc::Point\">Point</a>&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;BaseField = F&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["jf_plonk::circuit::customized::ecc::Point"]},{"text":"impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/circuit/customized/ecc/struct.Point.html\" title=\"struct jf_plonk::circuit::customized::ecc::Point\">Point</a>&lt;F&gt;&gt; for GroupAffine&lt;P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;BaseField = F&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["ark_ec::models::twisted_edwards_extended::GroupAffine"]},{"text":"impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/circuit/customized/ecc/struct.Point.html\" title=\"struct jf_plonk::circuit::customized::ecc::Point\">Point</a>&lt;F&gt;&gt; for GroupProjective&lt;P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;BaseField = F&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["ark_ec::models::twisted_edwards_extended::GroupProjective"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.array.html\">[</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.usize.html\">usize</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.array.html\">; 4]</a>&gt; for <a class=\"struct\" href=\"jf_plonk/circuit/customized/rescue/struct.RescueStateVar.html\" title=\"struct jf_plonk::circuit::customized::rescue::RescueStateVar\">RescueStateVar</a>","synthetic":false,"types":["jf_plonk::circuit::customized::rescue::native::RescueStateVar"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/circuit/struct.BoolVar.html\" title=\"struct jf_plonk::circuit::BoolVar\">BoolVar</a>&gt; for <a class=\"type\" href=\"jf_plonk/circuit/type.Variable.html\" title=\"type jf_plonk::circuit::Variable\">Variable</a>","synthetic":false,"types":["jf_plonk::circuit::Variable"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;Error&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>","synthetic":false,"types":["jf_plonk::errors::PlonkError"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://doc.rust-lang.org/1.63.0/std/io/error/struct.Error.html\" title=\"struct std::io::error::Error\">Error</a>&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>","synthetic":false,"types":["jf_plonk::errors::PlonkError"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;SerializationError&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>","synthetic":false,"types":["jf_plonk::errors::PlonkError"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_rescue/errors/enum.RescueError.html\" title=\"enum jf_rescue::errors::RescueError\">RescueError</a>&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>","synthetic":false,"types":["jf_plonk::errors::PlonkError"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_plonk/errors/enum.SnarkError.html\" title=\"enum jf_plonk::errors::SnarkError\">SnarkError</a>&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>","synthetic":false,"types":["jf_plonk::errors::PlonkError"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_plonk/errors/enum.CircuitError.html\" title=\"enum jf_plonk::errors::CircuitError\">CircuitError</a>&gt; for <a class=\"enum\" href=\"jf_plonk/errors/enum.PlonkError.html\" title=\"enum jf_plonk::errors::PlonkError\">PlonkError</a>","synthetic":false,"types":["jf_plonk::errors::PlonkError"]},{"text":"impl&lt;E, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.Proof.html\" title=\"struct jf_plonk::proof_system::structs::Proof\">Proof</a>&lt;E&gt;&gt; for <a class=\"struct\" href=\"https://doc.rust-lang.org/1.63.0/alloc/vec/struct.Vec.html\" title=\"struct alloc::vec::Vec\">Vec</a>&lt;E::Fq&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;E: PairingEngine&lt;G1Affine = GroupAffine&lt;P&gt;&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: SWModelParameters&lt;BaseField = E::Fq, ScalarField = E::Fr&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["alloc::vec::Vec"]},{"text":"impl&lt;E:&nbsp;PairingEngine&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.Proof.html\" title=\"struct jf_plonk::proof_system::structs::Proof\">Proof</a>&lt;E&gt;&gt; for <a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.BatchProof.html\" title=\"struct jf_plonk::proof_system::structs::BatchProof\">BatchProof</a>&lt;E&gt;","synthetic":false,"types":["jf_plonk::proof_system::structs::BatchProof"]},{"text":"impl&lt;F:&nbsp;Field&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.ProofEvaluations.html\" title=\"struct jf_plonk::proof_system::structs::ProofEvaluations\">ProofEvaluations</a>&lt;F&gt;&gt; for <a class=\"struct\" href=\"https://doc.rust-lang.org/1.63.0/alloc/vec/struct.Vec.html\" title=\"struct alloc::vec::Vec\">Vec</a>&lt;F&gt;","synthetic":false,"types":["alloc::vec::Vec"]},{"text":"impl&lt;E, F, P1, P2&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.VerifyingKey.html\" title=\"struct jf_plonk::proof_system::structs::VerifyingKey\">VerifyingKey</a>&lt;E&gt;&gt; for <a class=\"struct\" href=\"https://doc.rust-lang.org/1.63.0/alloc/vec/struct.Vec.html\" title=\"struct alloc::vec::Vec\">Vec</a>&lt;E::Fq&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;E: PairingEngine&lt;G1Affine = GroupAffine&lt;P1&gt;, G2Affine = GroupAffine&lt;P2&gt;, Fqe = Fp2&lt;F&gt;&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Fp2Parameters&lt;Fp = E::Fq&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;P1: SWModelParameters&lt;BaseField = E::Fq, ScalarField = E::Fr&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;P2: SWModelParameters&lt;BaseField = E::Fqe, ScalarField = E::Fr&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["alloc::vec::Vec"]}];
implementors["jf_primitives"] = [{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.array.html\">[</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.u8.html\">u8</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.array.html\">; 32]</a>&gt; for <a class=\"struct\" href=\"jf_primitives/aead/struct.EncKey.html\" title=\"struct jf_primitives::aead::EncKey\">EncKey</a>","synthetic":false,"types":["jf_primitives::aead::EncKey"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_primitives/aead/struct.EncKey.html\" title=\"struct jf_primitives::aead::EncKey\">EncKey</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.array.html\">[</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.u8.html\">u8</a><a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.array.html\">; 32]</a>","synthetic":false,"types":[]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_rescue/errors/enum.RescueError.html\" title=\"enum jf_rescue::errors::RescueError\">RescueError</a>&gt; for <a class=\"enum\" href=\"jf_primitives/errors/enum.PrimitivesError.html\" title=\"enum jf_primitives::errors::PrimitivesError\">PrimitivesError</a>","synthetic":false,"types":["jf_primitives::errors::PrimitivesError"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;SerializationError&gt; for <a class=\"enum\" href=\"jf_primitives/errors/enum.PrimitivesError.html\" title=\"enum jf_primitives::errors::PrimitivesError\">PrimitivesError</a>","synthetic":false,"types":["jf_primitives::errors::PrimitivesError"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_primitives/merkle_tree/enum.NodePos.html\" title=\"enum jf_primitives::merkle_tree::NodePos\">NodePos</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.usize.html\">usize</a>","synthetic":false,"types":[]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_primitives/merkle_tree/enum.NodePos.html\" title=\"enum jf_primitives::merkle_tree::NodePos\">NodePos</a>&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.u8.html\">u8</a>","synthetic":false,"types":[]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.u8.html\">u8</a>&gt; for <a class=\"enum\" href=\"jf_primitives/merkle_tree/enum.NodePos.html\" title=\"enum jf_primitives::merkle_tree::NodePos\">NodePos</a>","synthetic":false,"types":["jf_primitives::merkle_tree::NodePos"]},{"text":"impl&lt;F:&nbsp;PrimeField&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.u64.html\">u64</a>&gt; for <a class=\"struct\" href=\"jf_primitives/merkle_tree/struct.NodeValue.html\" title=\"struct jf_primitives::merkle_tree::NodeValue\">NodeValue</a>&lt;F&gt;","synthetic":false,"types":["jf_primitives::merkle_tree::NodeValue"]},{"text":"impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"jf_primitives/merkle_tree/enum.LookupResult.html\" title=\"enum jf_primitives::merkle_tree::LookupResult\">LookupResult</a>&lt;F, P&gt;&gt; for <a class=\"enum\" href=\"https://doc.rust-lang.org/1.63.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"enum\" href=\"https://doc.rust-lang.org/1.63.0/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.tuple.html\">(F, P)</a>&gt;&gt;","synthetic":false,"types":["core::option::Option"]},{"text":"impl&lt;F:&nbsp;<a class=\"trait\" href=\"jf_rescue/trait.RescueParameter.html\" title=\"trait jf_rescue::RescueParameter\">RescueParameter</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_primitives/merkle_tree/struct.FilledMTBuilder.html\" title=\"struct jf_primitives::merkle_tree::FilledMTBuilder\">FilledMTBuilder</a>&lt;F&gt;&gt; for <a class=\"struct\" href=\"jf_primitives/merkle_tree/struct.MerkleTree.html\" title=\"struct jf_primitives::merkle_tree::MerkleTree\">MerkleTree</a>&lt;F&gt;","synthetic":false,"types":["jf_primitives::merkle_tree::MerkleTree"]},{"text":"impl&lt;F:&nbsp;PrimeField&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;F&gt; for <a class=\"struct\" href=\"jf_primitives/prf/struct.PrfKey.html\" title=\"struct jf_primitives::prf::PrfKey\">PrfKey</a>&lt;F&gt;","synthetic":false,"types":["jf_primitives::prf::PrfKey"]},{"text":"impl&lt;P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;GroupAffine&lt;P&gt;&gt; for <a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.VerKey.html\" title=\"struct jf_primitives::signatures::schnorr::VerKey\">VerKey</a>&lt;P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["jf_primitives::signatures::schnorr::VerKey"]},{"text":"impl&lt;P, F&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.SignKey.html\" title=\"struct jf_primitives::signatures::schnorr::SignKey\">SignKey</a>&lt;F&gt;&gt; for <a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.VerKey.html\" title=\"struct jf_primitives::signatures::schnorr::VerKey\">VerKey</a>&lt;P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;ScalarField = F&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,&nbsp;</span>","synthetic":false,"types":["jf_primitives::signatures::schnorr::VerKey"]},{"text":"impl&lt;F, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.VerKey.html\" title=\"struct jf_primitives::signatures::schnorr::VerKey\">VerKey</a>&lt;P&gt;&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.tuple.html\">(F, F)</a> <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: PrimeField,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters&lt;BaseField = F&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"struct\" href=\"jf_primitives/elgamal/struct.EncKey.html\" title=\"struct jf_primitives::elgamal::EncKey\">EncKey</a>&lt;P&gt;&gt; for (P::BaseField, P::BaseField) <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Parameters + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,&nbsp;</span>","synthetic":false,"types":["ark_ec::models::ModelParameters"]}];
implementors["jf_rescue"] = [{"text":"impl&lt;F:&nbsp;<a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/marker/trait.Copy.html\" title=\"trait core::marker::Copy\">Copy</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.slice.html\">[F]</a>&gt; for <a class=\"struct\" href=\"jf_rescue/struct.RescueVector.html\" title=\"struct jf_rescue::RescueVector\">RescueVector</a>&lt;F&gt;","synthetic":false,"types":["jf_rescue::RescueVector"]},{"text":"impl&lt;F:&nbsp;<a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/marker/trait.Copy.html\" title=\"trait core::marker::Copy\">Copy</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.array.html\">[</a>F<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.array.html\">; 4]</a>&gt; for <a class=\"struct\" href=\"jf_rescue/struct.RescueVector.html\" title=\"struct jf_rescue::RescueVector\">RescueVector</a>&lt;F&gt;","synthetic":false,"types":["jf_rescue::RescueVector"]},{"text":"impl&lt;F:&nbsp;PrimeField&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.array.html\">[</a><a class=\"struct\" href=\"jf_rescue/struct.RescueVector.html\" title=\"struct jf_rescue::RescueVector\">RescueVector</a>&lt;F&gt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.array.html\">; 4]</a>&gt; for <a class=\"struct\" href=\"jf_rescue/struct.RescueMatrix.html\" title=\"struct jf_rescue::RescueMatrix\">RescueMatrix</a>&lt;F&gt;","synthetic":false,"types":["jf_rescue::RescueMatrix"]},{"text":"impl&lt;F:&nbsp;<a class=\"trait\" href=\"jf_rescue/trait.RescueParameter.html\" title=\"trait jf_rescue::RescueParameter\">RescueParameter</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"jf_rescue/struct.PRP.html\" title=\"struct jf_rescue::PRP\">PRP</a>&lt;F&gt;&gt; for <a class=\"struct\" href=\"jf_rescue/struct.Permutation.html\" title=\"struct jf_rescue::Permutation\">Permutation</a>&lt;F&gt;","synthetic":false,"types":["jf_rescue::Permutation"]}];
implementors["jf_utils"] = [{"text":"impl&lt;T:&nbsp;CanonicalSerialize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;T&gt; for <a class=\"struct\" href=\"jf_utils/struct.CanonicalBytes.html\" title=\"struct jf_utils::CanonicalBytes\">CanonicalBytes</a>","synthetic":false,"types":["jf_utils::serialize::CanonicalBytes"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"jf_utils/trait.Tagged.html\" title=\"trait jf_utils::Tagged\">Tagged</a> + CanonicalSerialize + CanonicalDeserialize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;T&gt; for <a class=\"struct\" href=\"jf_utils/struct.TaggedBlob.html\" title=\"struct jf_utils::TaggedBlob\">TaggedBlob</a>&lt;T&gt;","synthetic":false,"types":["jf_utils::serialize::TaggedBlob"]},{"text":"impl&lt;T:&nbsp;<a class=\"trait\" href=\"jf_utils/trait.Tagged.html\" title=\"trait jf_utils::Tagged\">Tagged</a> + CanonicalSerialize + CanonicalDeserialize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.63.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.63.0/std/primitive.reference.html\">&amp;</a>T&gt; for <a class=\"struct\" href=\"jf_utils/struct.TaggedBlob.html\" title=\"struct jf_utils::TaggedBlob\">TaggedBlob</a>&lt;T&gt;","synthetic":false,"types":["jf_utils::serialize::TaggedBlob"]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()