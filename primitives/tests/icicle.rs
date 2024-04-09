#[cfg(feature = "icicle")]
#[cfg(test)]
mod icicle_tests {
    use icicle_core::traits::FieldImpl;
    use jf_primitives::icicle_deps::*;

    type Scalar = icicle_bn254::curve::ScalarField;
    type Affine = icicle_bn254::curve::G1Affine;
    type Projective = icicle_bn254::curve::G1Projective;

    const BATCH_SIZE: usize = 8;
    const NUM_POINTS: usize = 8;

    #[test]
    fn icicle_batch_msm() {
        let stream = warmup_new_stream().unwrap();

        let mut cfg = MSMConfig::default();
        cfg.ctx.stream = &stream;
        cfg.is_async = true; // non-blocking

        let points = [Affine::zero(); NUM_POINTS];
        let mut points_on_device =
            HostOrDeviceSlice::<'_, Affine>::cuda_malloc(NUM_POINTS).unwrap();
        points_on_device.copy_from_host(&points).unwrap();

        let first_batch_scalars = [Scalar::one(); NUM_POINTS * BATCH_SIZE];
        let mut first_batch_scalars_on_device =
            HostOrDeviceSlice::<'_, Scalar>::cuda_malloc(NUM_POINTS * BATCH_SIZE).unwrap();
        first_batch_scalars_on_device
            .copy_from_host(&first_batch_scalars)
            .unwrap();

        let mut msm_result = HostOrDeviceSlice::<'_, Projective>::cuda_malloc(BATCH_SIZE).unwrap();

        icicle_core::msm::msm(
            &first_batch_scalars_on_device,
            &points_on_device,
            &cfg,
            &mut msm_result,
        )
        .unwrap();

        stream.synchronize().unwrap();

        let mut msm_host_result = vec![Projective::zero(); BATCH_SIZE];
        msm_result.copy_to_host(&mut msm_host_result[..]).unwrap();

        ark_std::println!("First run should be successful.");

        let second_batch_scalars = [Scalar::one(); NUM_POINTS * BATCH_SIZE];
        let mut second_batch_scalars_on_device =
            HostOrDeviceSlice::<'_, Scalar>::cuda_malloc(NUM_POINTS * BATCH_SIZE).unwrap();
        second_batch_scalars_on_device
            .copy_from_host(&second_batch_scalars)
            .unwrap();

        let mut msm_result = HostOrDeviceSlice::<'_, Projective>::cuda_malloc(BATCH_SIZE).unwrap();

        icicle_core::msm::msm(
            &second_batch_scalars_on_device,
            &points_on_device,
            &cfg,
            &mut msm_result,
        )
        .unwrap();

        stream.synchronize().unwrap();

        let mut msm_host_result = vec![Projective::zero(); BATCH_SIZE];
        msm_result.copy_to_host(&mut msm_host_result[..]).unwrap();

        ark_std::println!("Second run should be successful.");

        let third_batch_scalars = [Scalar::one(); NUM_POINTS * (BATCH_SIZE - 1)];
        let mut third_batch_scalars_on_device =
            HostOrDeviceSlice::<'_, Scalar>::cuda_malloc(NUM_POINTS * (BATCH_SIZE - 1)).unwrap();
        third_batch_scalars_on_device
            .copy_from_host(&third_batch_scalars)
            .unwrap();

        let mut msm_result =
            HostOrDeviceSlice::<'_, Projective>::cuda_malloc(BATCH_SIZE - 1).unwrap();

        icicle_core::msm::msm(
            &third_batch_scalars_on_device,
            &points_on_device,
            &cfg,
            &mut msm_result,
        )
        .unwrap();

        stream.synchronize().unwrap();

        let mut msm_host_result = vec![Projective::zero(); BATCH_SIZE - 1];
        msm_result.copy_to_host(&mut msm_host_result[..]).unwrap();

        ark_std::println!("Third run should be successful.");

        let fourth_batch_scalars = [Scalar::one(); NUM_POINTS * (BATCH_SIZE + 1)];
        let mut fourth_batch_scalars_on_device =
            HostOrDeviceSlice::<'_, Scalar>::cuda_malloc(NUM_POINTS * (BATCH_SIZE + 1)).unwrap();
        fourth_batch_scalars_on_device
            .copy_from_host(&fourth_batch_scalars)
            .unwrap();

        let mut msm_result =
            HostOrDeviceSlice::<'_, Projective>::cuda_malloc(BATCH_SIZE + 1).unwrap();

        icicle_core::msm::msm(
            &fourth_batch_scalars_on_device,
            &points_on_device,
            &cfg,
            &mut msm_result,
        )
        .unwrap();

        stream.synchronize().unwrap();

        let mut msm_host_result = vec![Projective::zero(); BATCH_SIZE + 1];
        msm_result.copy_to_host(&mut msm_host_result[..]).unwrap();

        ark_std::println!("Fourth run should fail, we shouldn't see this.");
    }
}
