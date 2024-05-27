#![no_std]

use ark_std::{borrow::Borrow, fmt::Debug, hash::Hash, error::Error, UniformRand};

/// Result type for cryptographic verification to improve code semantics and error handling.
type VerificationResult = Result<(), ()>;

/// Defines a cryptographic commitment scheme with generic types for inputs, outputs, and randomness.
///
/// This trait provides methods for creating and verifying commitments. Implementations must specify
/// how commitments are generated and verified, including handling of optional randomness for blinding.
pub trait CommitmentScheme {
    /// The type of input data used to create a commitment.
    type Input;
    /// The type of the commitment output, must support cloning, hashing, and comparison.
    type Output: Clone + Debug + PartialEq + Eq + Hash;
    /// The type for the randomness used in the commitment process, supporting uniform random generation.
    type Randomness: Clone + Debug + PartialEq + Eq + UniformRand;
    /// The type of error that may be returned by commitment operations.
    type Error: Error;

    /// Creates a commitment from the given `input` and an optional `randomness`.
    ///
    /// The `randomness` is optional and can be used to blind the commitment if required by the scheme.
    /// Returns either the commitment or an error if the operation cannot be completed.
    fn commit<T: Borrow<Self::Input>>(
        input: T,
        randomness: Option<&Self::Randomness>,
    ) -> Result<Self::Output, Self::Error>;

    /// Verifies a commitment against the original `input` and the provided `randomness`.
    ///
    /// This method checks if the given `commitment` corresponds to the `input` with the optional `randomness`.
    /// It returns `Ok` wrapped in `VerificationResult` if the verification is successful, otherwise an error.
    fn verify<T: Borrow<Self::Input>>(
        input: T,
        randomness: Option<&Self::Randomness>,
        commitment: &Self::Output,
    ) -> Result<VerificationResult, Self::Error>;
}
