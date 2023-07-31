(function() {var implementors = {
"jf_plonk":[["impl&lt;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + Field&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.ProofEvaluations.html\" title=\"struct jf_plonk::proof_system::structs::ProofEvaluations\">ProofEvaluations</a>&lt;F&gt;"],["impl&lt;E&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.Proof.html\" title=\"struct jf_plonk::proof_system::structs::Proof\">Proof</a>&lt;E&gt;<span class=\"where fmt-newline\">where\n    E: Pairing,</span>"],["impl&lt;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + Field&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.PlookupEvaluations.html\" title=\"struct jf_plonk::proof_system::structs::PlookupEvaluations\">PlookupEvaluations</a>&lt;F&gt;"],["impl&lt;E&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.BatchProof.html\" title=\"struct jf_plonk::proof_system::structs::BatchProof\">BatchProof</a>&lt;E&gt;<span class=\"where fmt-newline\">where\n    E: Pairing,</span>"],["impl&lt;E&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_plonk/proof_system/structs/struct.PlookupProof.html\" title=\"struct jf_plonk::proof_system::structs::PlookupProof\">PlookupProof</a>&lt;E&gt;<span class=\"where fmt-newline\">where\n    E: Pairing,</span>"]],
"jf_primitives":[["impl&lt;P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.VerKey.html\" title=\"struct jf_primitives::signatures::schnorr::VerKey\">VerKey</a>&lt;P&gt;<span class=\"where fmt-newline\">where\n    P: Config,</span>"],["impl&lt;E, H, Arity, N, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/merkle_tree/namespaced_merkle_tree/struct.NMT.html\" title=\"struct jf_primitives::merkle_tree::namespaced_merkle_tree::NMT\">NMT</a>&lt;E, H, Arity, N, T&gt;<span class=\"where fmt-newline\">where\n    H: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.DigestAlgorithm.html\" title=\"trait jf_primitives::merkle_tree::DigestAlgorithm\">DigestAlgorithm</a>&lt;E, <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.0/std/primitive.u64.html\">u64</a>, T&gt; + <a class=\"trait\" href=\"jf_primitives/merkle_tree/namespaced_merkle_tree/trait.BindNamespace.html\" title=\"trait jf_primitives::merkle_tree::namespaced_merkle_tree::BindNamespace\">BindNamespace</a>&lt;E, <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.0/std/primitive.u64.html\">u64</a>, T, N&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    E: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.Element.html\" title=\"trait jf_primitives::merkle_tree::Element\">Element</a> + <a class=\"trait\" href=\"jf_primitives/merkle_tree/namespaced_merkle_tree/trait.Namespaced.html\" title=\"trait jf_primitives::merkle_tree::namespaced_merkle_tree::Namespaced\">Namespaced</a>&lt;Namespace = N&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    T: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.NodeValue.html\" title=\"trait jf_primitives::merkle_tree::NodeValue\">NodeValue</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    N: <a class=\"trait\" href=\"jf_primitives/merkle_tree/namespaced_merkle_tree/trait.Namespace.html\" title=\"trait jf_primitives::merkle_tree::namespaced_merkle_tree::Namespace\">Namespace</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    Arity: <a class=\"trait\" href=\"https://docs.rs/typenum/1.16.0/typenum/marker_traits/trait.Unsigned.html\" title=\"trait typenum::marker_traits::Unsigned\">Unsigned</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,</span>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/merkle_tree/examples/struct.Sha3Digest.html\" title=\"struct jf_primitives::merkle_tree::examples::Sha3Digest\">Sha3Digest</a>"],["impl&lt;P: Config&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/elgamal/struct.EncKey.html\" title=\"struct jf_primitives::elgamal::EncKey\">EncKey</a>&lt;P&gt;"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/merkle_tree/examples/struct.Sha3Node.html\" title=\"struct jf_primitives::merkle_tree::examples::Sha3Node\">Sha3Node</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/signatures/bls_over_bn254/struct.VerKey.html\" title=\"struct jf_primitives::signatures::bls_over_bn254::VerKey\">VerKey</a>"],["impl&lt;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + PrimeField&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.SignKey.html\" title=\"struct jf_primitives::signatures::schnorr::SignKey\">SignKey</a>&lt;F&gt;"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/signatures/bls_over_bn254/struct.SignKey.html\" title=\"struct jf_primitives::signatures::bls_over_bn254::SignKey\">SignKey</a>"],["impl&lt;E, H, I, Arity, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/merkle_tree/universal_merkle_tree/struct.UniversalMerkleTree.html\" title=\"struct jf_primitives::merkle_tree::universal_merkle_tree::UniversalMerkleTree\">UniversalMerkleTree</a>&lt;E, H, I, Arity, T&gt;<span class=\"where fmt-newline\">where\n    E: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.Element.html\" title=\"trait jf_primitives::merkle_tree::Element\">Element</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    H: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.DigestAlgorithm.html\" title=\"trait jf_primitives::merkle_tree::DigestAlgorithm\">DigestAlgorithm</a>&lt;E, I, T&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    I: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.Index.html\" title=\"trait jf_primitives::merkle_tree::Index\">Index</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    Arity: <a class=\"trait\" href=\"https://docs.rs/typenum/1.16.0/typenum/marker_traits/trait.Unsigned.html\" title=\"trait typenum::marker_traits::Unsigned\">Unsigned</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    T: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.NodeValue.html\" title=\"trait jf_primitives::merkle_tree::NodeValue\">NodeValue</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,</span>"],["impl&lt;H&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/merkle_tree/hasher/struct.HasherNode.html\" title=\"struct jf_primitives::merkle_tree::hasher::HasherNode\">HasherNode</a>&lt;H&gt;<span class=\"where fmt-newline\">where\n    H: Digest,</span>"],["impl&lt;P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/elgamal/struct.Ciphertext.html\" title=\"struct jf_primitives::elgamal::Ciphertext\">Ciphertext</a>&lt;P&gt;<span class=\"where fmt-newline\">where\n    P: Config,</span>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/signatures/bls_over_bn254/struct.Signature.html\" title=\"struct jf_primitives::signatures::bls_over_bn254::Signature\">Signature</a>"],["impl&lt;E, H, I, Arity, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/merkle_tree/light_weight/struct.LightWeightMerkleTree.html\" title=\"struct jf_primitives::merkle_tree::light_weight::LightWeightMerkleTree\">LightWeightMerkleTree</a>&lt;E, H, I, Arity, T&gt;<span class=\"where fmt-newline\">where\n    E: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.Element.html\" title=\"trait jf_primitives::merkle_tree::Element\">Element</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    H: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.DigestAlgorithm.html\" title=\"trait jf_primitives::merkle_tree::DigestAlgorithm\">DigestAlgorithm</a>&lt;E, I, T&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    I: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.Index.html\" title=\"trait jf_primitives::merkle_tree::Index\">Index</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    Arity: <a class=\"trait\" href=\"https://docs.rs/typenum/1.16.0/typenum/marker_traits/trait.Unsigned.html\" title=\"trait typenum::marker_traits::Unsigned\">Unsigned</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    T: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.NodeValue.html\" title=\"trait jf_primitives::merkle_tree::NodeValue\">NodeValue</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,</span>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/aead/struct.Ciphertext.html\" title=\"struct jf_primitives::aead::Ciphertext\">Ciphertext</a>"],["impl&lt;E: Pairing&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/pcs/prelude/struct.Commitment.html\" title=\"struct jf_primitives::pcs::prelude::Commitment\">Commitment</a>&lt;E&gt;"],["impl&lt;E, H, I, Arity, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/merkle_tree/append_only/struct.MerkleTree.html\" title=\"struct jf_primitives::merkle_tree::append_only::MerkleTree\">MerkleTree</a>&lt;E, H, I, Arity, T&gt;<span class=\"where fmt-newline\">where\n    E: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.Element.html\" title=\"trait jf_primitives::merkle_tree::Element\">Element</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    H: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.DigestAlgorithm.html\" title=\"trait jf_primitives::merkle_tree::DigestAlgorithm\">DigestAlgorithm</a>&lt;E, I, T&gt; + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    I: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.Index.html\" title=\"trait jf_primitives::merkle_tree::Index\">Index</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    Arity: <a class=\"trait\" href=\"https://docs.rs/typenum/1.16.0/typenum/marker_traits/trait.Unsigned.html\" title=\"trait typenum::marker_traits::Unsigned\">Unsigned</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,\n    T: <a class=\"trait\" href=\"jf_primitives/merkle_tree/trait.NodeValue.html\" title=\"trait jf_primitives::merkle_tree::NodeValue\">NodeValue</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>,</span>"],["impl&lt;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>, P: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>, N: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"enum\" href=\"jf_primitives/merkle_tree/enum.LookupResult.html\" title=\"enum jf_primitives::merkle_tree::LookupResult\">LookupResult</a>&lt;F, P, N&gt;"],["impl&lt;P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/signatures/schnorr/struct.Signature.html\" title=\"struct jf_primitives::signatures::schnorr::Signature\">Signature</a>&lt;P&gt;<span class=\"where fmt-newline\">where\n    P: Config,</span>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"jf_primitives/aead/struct.EncKey.html\" title=\"struct jf_primitives::aead::EncKey\">EncKey</a>"]],
"jf_relation":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"enum\" href=\"jf_relation/constraint_system/enum.PlonkType.html\" title=\"enum jf_relation::constraint_system::PlonkType\">PlonkType</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> for <a class=\"enum\" href=\"jf_relation/constraint_system/enum.MergeableCircuitType.html\" title=\"enum jf_relation::constraint_system::MergeableCircuitType\">MergeableCircuitType</a>"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()