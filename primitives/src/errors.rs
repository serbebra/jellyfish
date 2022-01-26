//! Error types.

use ark_std::string::String;
use displaydoc::Display;
use jf_rescue::errors::RescueError;

/// A `enum` specifying the possible failure modes of the primitives.
#[derive(Debug, Display)]
pub enum PrimitivesError {
    /// Unsuccessful verification for proof or signature, {0}
    VerificationError(String),
    /// Bad parameter in function call, {0}
    ParameterError(String),
    #[rustfmt::skip]
    /// ‼ ️Internal error! Please report to Crypto Team immediately!\nMessage: {0}
    InternalError(String),
    /// Deserialization failed: {0}
    DeserializationError(String),
    /// Decryption failed: {0}
    FailedDecryption(String),
    /// Rescue Error: {0}
    RescueError(RescueError),
    /// Inconsistent Structure error, {0}
    InconsistentStructureError(String),
}

impl From<RescueError> for PrimitivesError {
    fn from(e: RescueError) -> Self {
        Self::RescueError(e)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PrimitivesError {}
