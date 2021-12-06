import SimpleITK as sitk
import itk

from collections import namedtuple
from scipy.stats._discrete_distns import poisson

from scipy import stats

import numpy as np


def sitk_to_itk(sitk_image):
    """
    Helper function to convert SimpleITK images to ITK images
    """
    sitk_arr = sitk.GetArrayFromImage(sitk_image)

    itk_image = itk.GetImageFromArray(sitk_arr, is_vector=False)
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())
    itk_image.SetDirection(
        itk.GetMatrixFromArray(np.reshape(np.array(sitk_image.GetDirection()), [3] * 2))
    )

    return itk_image


def itk_to_sitk(itk_image):
    """
    Helper function to convert ITK images to SimpleITK images
    """
    sitk_image = sitk.GetImageFromArray(
        itk.GetArrayFromImage(itk_image), isVector=False
    )
    sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    sitk_image.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten())

    return sitk_image


def morphological_interpolate(sitk_image):
    """
    Performs morphological interpolation
    See: https://github.com/KitwareMedical/ITKMorphologicalContourInterpolation

    Useful for filling in gaps in contouring between slices
    """

    itk_image = sitk_to_itk(sitk_image)

    output_type = itk.Image[itk.US, 3]

    f_cast = itk.CastImageFilter[itk_image, output_type].New()
    f_cast.SetInput(itk_image)
    img_cast = f_cast.GetOutput()

    f_interpolator = itk.MorphologicalContourInterpolator.New()
    f_interpolator.SetInput(img_cast)
    f_interpolator.Update()

    img_interpolated = f_interpolator.GetOutput()

    sitk_img_interpolated = itk_to_sitk(img_interpolated)

    return sitk_img_interpolated


def interpolate_image(image, as_integer=True):
    """
    Cell density maps/Histology images
    Fill in missing slices
    """
    arr = sitk.GetArrayFromImage(image)

    # Get indices where image is defined
    slice_indices = np.unique(np.where(arr)[0])

    # Get missing indices
    fill_indices = (slice_indices[1:] + slice_indices[:-1]) // 2

    for i in fill_indices:
        if as_integer:
            arr[i] = arr[i - 1] // 2 + arr[i + 1] // 2
        else:
            arr[i] = arr[i - 1] / 2 + arr[i + 1] / 2

    # End slices
    i_min = np.min(slice_indices)
    arr[i_min - 1] = (arr[i_min]) // 2

    i_max = np.max(slice_indices)
    arr[i_max + 1] = (arr[i_max]) // 2

    image_interp = sitk.GetImageFromArray(arr)
    image_interp.CopyInformation(image)

    return image_interp


def interpolate_histology_lesion_probability(histology_mask_image):
    """
    Histology slices are only defined every 5mm
    Registered to 2.5mm space MRI, so adjacent slices are weighted with 0.5
    """
    total_lesion = histology_mask_image > 0

    if sitk.GetArrayFromImage(total_lesion).sum() == 0:
        # There is no data, return zeros
        return histology_mask_image

    total_lesion = sitk.Cast(total_lesion, sitk.sitkFloat32)

    arr_histology = sitk.GetArrayFromImage(total_lesion)

    # Get indices where cell density is defined
    slice_indices = np.unique(np.where(arr_histology)[0])

    # Get missing indices
    fill_indices = (slice_indices[1:] + slice_indices[:-1]) // 2

    for i in fill_indices:
        arr_histology[i] = (arr_histology[i - 1]) / 2 + (arr_histology[i + 1]) / 2

    # End slices
    i_min = np.min(slice_indices)
    arr_histology[i_min - 1] = (arr_histology[i_min]) / 2

    i_max = np.max(slice_indices)
    arr_histology[i_max + 1] = (arr_histology[i_max]) / 2

    lesion_probability = sitk.GetImageFromArray(arr_histology)
    lesion_probability.CopyInformation(total_lesion)

    return lesion_probability


def generate_sampling_label(histology_image):
    """
    Generate a binary mask for where the histology slices are
    This will get deformed later so we need to keep track of it
    """
    image = sitk.VectorIndexSelectionCast(histology_image, 0, sitk.sitkFloat32)

    arr = sitk.GetArrayFromImage(image)

    # Get indices where image is defined
    slice_indices = np.unique(np.where(arr)[0])

    # Get missing indices
    fill_indices = (slice_indices[1:] + slice_indices[:-1]) // 2

    for i in slice_indices:
        arr[i] = arr[0] * 0 + 1

    for i in fill_indices:
        arr[i] = arr[0] * 0 + 1

    # End slices
    i_min = np.min(slice_indices)
    arr[i_min - 1] = 0.5 * (arr[0] * 0 + 1)

    i_max = np.max(slice_indices)
    arr[i_max + 1] = 0.5 * (arr[0] * 0 + 1)

    image_sampling_label = sitk.GetImageFromArray(arr)
    image_sampling_label.CopyInformation(image)

    return image_sampling_label


def poisson_means_test(
    count1, nobs1, count2, nobs2, diff=0, alternative="two-sided", return_only_p=False
):
    r"""
    Calculate E-test for the mean difference of two samples that follow a Poisson
    distribution from descriptive statistics
    Let :math:`X_{11},...,X_{1n_1}` and :math:`X_{21},...,X_{2n_2}` be independent
    samples respectively, from :math:`Poisson(\lambda_1)` and :math:`Poisson(\lambda_2)` distributions. It is well known that

    .. math:: X_1 = \sum_{i=1}^{n_1} X_{1i} \sim Poisson(n_1\lambda_1)

    independently of

    .. math:: X_2 = \sum_{i=1}^{n_2} X_{2i} \sim Poisson(n_2\lambda_2)

    Let `count1` and `count1` be the observed values of :math:`X_1` and :math:`X_2`,
    respectively. The problem of interest here is to test

    .. math::
       H_0: \lambda_1 - \lambda_2 \le \mathtt{diff} \quad vs. \quad
       H_a: \lambda_1 - \lambda_2 > \mathtt{diff}

    for right sided `greater` hypothesis where :math:`\mathtt{diff} \ge 0` is a given
    number, based on (`nobs1`, `count1`, `nobs2`, `count2`). `two-sided` and `greater`
    cases are demonstrated by [1]_.The `less` hypothesis performed by switching the
    arguments on `greater` hypothesis.
    Parameters
    ----------
    count1, count2 : int
        Count event of interest from the first and second samples respectively
    nobs1, nobs2 : int
        Sample size observed
    diff : int of float, optional
        The difference of mean between two samples under null hypothesis
    alternative : {'two-sided', 'less', 'greater'}, optional
        Which alternative hypothesis to test
    Returns
    -------
    statistic : float
        The test statistic calculated from observed samples
    pvalue : float
        The associated p-value based on estimated p-value of the standardized
        difference.
    Notes
    -----
    The Poisson distribution is commonly used to model many processes such as
    transactions per user. A simple test to compare difference between two
    means of Poisson samples is C-test, based on conditional distribution.
    Meanwhile the E-test is an unconditional test
    Based the author results [1]_, E-test is more powerful than C-test. The E-test
    is almost exact because the test exceed the nominal value only by negligible
    amount. Compared to C-test which produce smaller size than nominal value.
    References
    ----------
    .. [1]  Krishnamoorthy, K., & Thomson, J. (2004). A more powerful test for
       comparing two Poisson means. Journal of Statistical Planning and Inference,
       119(1), 23-35.
    .. [2]  Przyborowski, J., & Wilenski, H. (1940). Homogeneity of results in
       testing samples from Poisson series: With an application to testing clover
       seed for dodder. Biometrika, 31(3/4), 313-323.
    Examples
    --------
    >>> from scipy import stats
    Taken from Przyborowski and Wilenski (1940) [2]_. Suppose that a purchaser wishes to
    test the number of dodder seeds (a weed) in a sack of clover seeds that he bought
    from a seed manufacturing company. A 100 g sample is drawn from a sack of clover
    seeds prior to being shipped to the purchaser. The sample is analyzed and found to
    contain no dodder seeds; that is, k1 = 0. Upon arrival, the purchaser also draws
    a 100 g sample from the sack. This time, three dodder seeds are found in the sample;
    that is, k2 = 3. The purchaser wishes to determine if the difference between the
    samples could not be due to chance.
    >>> stats.poisson_means_test(0, 100, 3, 100)
    PoissonMeansTestResult(statistic=-1.7320508075688772, pvalue=0.08837900929018155)
    >>> stats.ttest_ind_from_stats(mean1=0, std1=0, nobs1=100, equal_var=False,
    ...                            mean2=3.0/100, std2=np.sqrt(.03), nobs2=100)
    Ttest_indResult(statistic=-1.7320508075688772, pvalue=0.0863790757063214)
    The result above show evidence that the difference between two samples is significant
    at the level of significance 0.1. Both t-test and e-test show similar p-value, but
    will be different if the sample size is small
    >>> stats.poisson_means_test(0, 10, 3, 10)
    PoissonMeansTestResult(statistic=-1.7320508075688772, pvalue=0.08837900929018155)
    >>> stats.ttest_ind_from_stats(mean1=0, std1=0, nobs1=10, equal_var=False,
    ...                            mean2=3.0/10, std2=np.sqrt(.3), nobs2=10)
    Ttest_indResult(statistic=-1.7320508075688774, pvalue=0.11730680301423814)
    The t-test need n=100 to arrive at the similar p-value of e-test whereas the e-test
    produce similar p-value either with nobs=10 or nobs=100
    """

    PoissonMeansTestResult = namedtuple(
        "PoissonMeansTestResult", ("statistic", "pvalue")
    )

    if (
        any(not isinstance(item, int) for item in [count1, count2, nobs1, nobs2])
        and any(not isinstance(item, np.int) for item in [count1, count2, nobs1, nobs2])
        and any(
            not isinstance(item, np.int64) for item in [count1, count2, nobs1, nobs2]
        )
    ):
        print([type(count1), type(count2), type(nobs1), type(nobs2)])
        raise TypeError("int arguments required for count1, count2, nobs1, and nobs2")

    if count1 < 0 or count2 < 0:
        raise ValueError("k1 and k2 should have values greater than or equal to 0")

    if nobs1 <= 0 or nobs2 <= 0:
        print(nobs1, nobs2)
        raise ValueError("n1 and n2 should have values greater than 0")

    if diff < 0:
        raise ValueError("diff can not have negative values")

    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(
            "alternative should be one of {'two-sided', 'less', 'greater'}"
        )

    # reverse the arguments of sample one and sample two if the hypothesis selected
    # is `less`
    if alternative == "less":
        count1, count2 = count2, count1
        nobs1, nobs2 = nobs2, nobs1

    lmbd_hat2 = (count1 + count2) / (nobs1 + nobs2) - diff * nobs1 / (nobs1 + nobs2)

    # based on paper explanation, we do not need to calculate p-value if the `lmbd_hat2`
    # has value less than or equals zero, see Reference 1 page 26 below eq. 3.6
    if lmbd_hat2 <= 0:
        if return_only_p:
            return 1
        return PoissonMeansTestResult(None, 1)

    var = count1 / nobs1 ** 2 + count2 / nobs2 ** 2

    t_k1k2 = (count1 / nobs1 - count2 / nobs2 - diff) / np.sqrt(var)

    nlmbd_hat1 = nobs1 * (lmbd_hat2 + diff)
    nlmbd_hat2 = nobs2 * lmbd_hat2

    x1_lb, x1_ub = poisson.ppf([1e-10, 1 - 1e-10], nlmbd_hat1)
    x2_lb, x2_ub = poisson.ppf([1e-10, 1 - 1e-10], nlmbd_hat2)

    x1 = np.repeat(np.arange(x1_lb, x1_ub + 1), x2_ub - x2_lb + 1)
    x2 = np.resize(np.arange(x2_lb, x2_ub + 1), len(x1))

    prob_x1 = poisson.pmf(x1, nlmbd_hat1)
    prob_x2 = poisson.pmf(x2, nlmbd_hat2)

    lmbd_hat_x1 = x1 / nobs1
    lmbd_hat_x2 = x2 / nobs2

    diff_lmbd_x1x2 = lmbd_hat_x1 - lmbd_hat_x2 - diff
    var_x1x2 = lmbd_hat_x1 / nobs1 + lmbd_hat_x2 / nobs2

    if alternative == "two-sided":
        t_x1x2 = np.divide(
            diff_lmbd_x1x2,
            np.sqrt(var_x1x2),
            out=np.zeros_like(diff_lmbd_x1x2),
            where=(np.abs(lmbd_hat_x1 - lmbd_hat_x2) > diff),
        )
        p_x1x2 = np.multiply(
            prob_x1,
            prob_x2,
            out=np.zeros_like(prob_x1),
            where=(np.abs(t_x1x2) >= np.abs(t_k1k2)),
        )
    else:
        t_x1x2 = np.divide(
            diff_lmbd_x1x2,
            np.sqrt(var_x1x2),
            out=np.zeros_like(diff_lmbd_x1x2),
            where=(diff_lmbd_x1x2 > 0),
        )
        p_x1x2 = np.multiply(
            prob_x1, prob_x2, out=np.zeros_like(prob_x1), where=(t_x1x2 >= t_k1k2)
        )

    pvalue = np.sum(p_x1x2)

    if return_only_p:
        return pvalue

    return PoissonMeansTestResult(t_k1k2, pvalue)


def calculate_sig_map(
    img_values_1,
    img_sampling_1,
    img_values_2,
    img_sampling_2,
    img_sigma_1=None,
    img_sigma_2=None,
    img_val_as_counts=False,
    sigma_as_sqrt_counts=False,
    test="student_t",
):
    """Generate a 3D SimpleITK image with p-values
    representing statistically significant differences between
    img_values_1 and img_values_2

    Args:
        img_values_1 (sitk.Image): Sampled values. Can be counts or a real-valued (continuous) variable.
        img_sampling_1 (sitk.Image): Sampling frequency. Should be an integer (but will be cast to one anyway).
        img_values_2 (sitk.Image): Sampled values. Can be counts or a real-valued (continuous) variable.
        img_sampling_2 (sitk.Image): Sampling frequency. Should be an integer (but will be cast to one anyway).
        img_sigma_1 (sitk.Image, optional): Sample standard deviation. Defaults to None, in which case
                                            it will be estimated from img_values_1.
        img_sigma_2 (sitk.Image, optional): Sample standard deviation. Defaults to None, in which case
                                            it will be estimated from img_values_1.
        img_val_as_counts (bool, optional): Do the img_values represent counts? Defaults to False.
        sigma_as_sqrt_counts (bool, optional): Should we estimate the sample standard deviation as
                                               sqrt(K)? If samples come from a Poisson distribution
                                               you probably want to do this. Defaults to False.
        test (str, optional): The statistical test. Choose from "student_t" and "poisson_e". Defaults to "student_t".

    Returns:
       sitk.Image : The map of statistical significance.
    """

    arr_sampling_1 = sitk.GetArrayFromImage(img_sampling_1).astype(int)
    arr_sampling_2 = sitk.GetArrayFromImage(img_sampling_2).astype(int)

    arr_values_1 = sitk.GetArrayFromImage(img_values_1)
    arr_values_2 = sitk.GetArrayFromImage(img_values_2)

    if test == "poisson_e" or img_val_as_counts:
        # We need to ensure data it represented correctly
        arr_values_1 = np.ceil(arr_values_1).astype(int)
        arr_values_2 = np.ceil(arr_values_2).astype(int)

        arr_values_1[arr_values_1 < 0] = 0
        arr_values_2[arr_values_2 < 0] = 0

    # Flatten
    n_obs_1 = arr_sampling_1.flatten()
    n_obs_2 = arr_sampling_2.flatten()

    vals_1 = arr_values_1.flatten()
    vals_2 = arr_values_2.flatten()

    # Reduce to important samples
    # we don't need to test everything if there's no data!
    valid_indices = (n_obs_1 > 0) * (n_obs_2 > 0)

    # Extract samples
    n_obs_1 = n_obs_1[valid_indices]
    n_obs_2 = n_obs_2[valid_indices]
    vals_1 = vals_1[valid_indices]
    vals_2 = vals_2[valid_indices]

    if test == "poisson_e":
        f_test_vect = np.vectorize(poisson_means_test, otypes=[float])

        pval_out = f_pmt(vals_1, n_obs_1, vals_2, n_obs_2, return_only_p=True)

    elif test == "student_t":
        f_test_vect = np.vectorize(stats.ttest_ind_from_stats)

        n_obs_1_p = n_obs_1[:]
        n_obs_1_p[n_obs_1 <= 1] = 2

        n_obs_2_p = n_obs_2[:]
        n_obs_2_p[n_obs_2 <= 1] = 2

        if sigma_as_sqrt_counts:

            sigma_1 = np.sqrt(vals_1)
            sigma_1[vals_1 == 0] = 1

            sigma_2 = np.sqrt(vals_2)
            sigma_2[vals_2 == 0] = 1

            # Run statistical tests
            # This can take ~few mins
            results = f_test_vect(
                mean1=vals_1,
                std1=sigma_1,
                nobs1=n_obs_1_p,
                equal_var=False,
                mean2=vals_2,
                std2=sigma_2,
                nobs2=n_obs_2_p,
            )

        else:
            sigma_1 = sitk.GetArrayFromImage(img_sigma_1).flatten()
            sigma_1 = sigma_1[valid_indices]
            sigma_1[vals_1 == 0] = 1

            sigma_2 = sitk.GetArrayFromImage(img_sigma_2).flatten()
            sigma_2 = sigma_2[valid_indices]
            sigma_2[vals_1 == 0] = 1

            sigma_1[sigma_1 == 0] = 1e-3
            sigma_2[sigma_2 == 0] = 1e-3

            # Run statistical tests
            # This can take ~few mins
            results = f_test_vect(
                mean1=vals_1,
                std1=sigma_1,
                nobs1=n_obs_1,
                equal_var=False,
                mean2=vals_2,
                std2=sigma_2,
                nobs2=n_obs_2,
            )

        pval_out = results[1]

    sig_array = np.ones_like(arr_sampling_1.flatten(), dtype=float)
    sig_array[valid_indices] = pval_out

    sig_image = sitk.GetImageFromArray(sig_array.reshape(arr_sampling_1.shape))
    sig_image.CopyInformation(img_values_1)

    return sig_image


def smooth_edges(image, mask, dilate_voxels=1, smooth_mm=1):

    boundary = sitk.BinaryDilate(mask, (dilate_voxels,) * 3) - mask

    smooth_edge = sitk.Cast(
        sitk.Mask(sitk.SmoothingRecursiveGaussian(image, (smooth_mm,) * 3), boundary),
        image.GetPixelID(),
    )

    inside = sitk.MaskNegated(image, sitk.Cast(boundary, image.GetPixelID()))

    return sitk.Mask(smooth_edge + inside, mask)


def read_atlas_data(case_id_list, reg_types, label_names, image_names, input_dir):
    atlas_set = {}

    atlas_id_list = case_id_list[:]

    for atlas_id in atlas_id_list:
        atlas_set[atlas_id] = {}

        for reg_type in reg_types:
            atlas_set[atlas_id][reg_type] = {}

            for image_name in image_names:
                atlas_set[atlas_id][reg_type][image_name] = sitk.ReadImage(
                    (
                        input_dir
                        / f"MRHIST{atlas_id}"
                        / f"IMAGES_{reg_type}"
                        / f"MRHIST{atlas_id}_{image_name}.nii.gz"
                    ).as_posix(),
                    sitk.sitkFloat64,
                )

            for label_name in label_names:
                atlas_set[atlas_id][reg_type][label_name] = sitk.ReadImage(
                    (
                        input_dir
                        / f"MRHIST{atlas_id}"
                        / f"LABELS_{reg_type}"
                        / f"MRHIST{atlas_id}_{label_name}.nii.gz"
                    ).as_posix(),
                    sitk.sitkFloat64,
                )

    return atlas_set


def kl_divergence(normal_model, empirical_model, x=np.linspace(0, 100000, 101)):
    """
    Compute the Kullback-Leibler divergence
    Returns the relative entropy gained by using the empirical_model over the normal model

    Bandwidth computed automatically
    """

    q_norm = normal_model.pdf(x)
    p_norm = empirical_model(x)

    integrand = p_norm * np.log2(p_norm / q_norm)
    integrand[p_norm == 0] = 0

    rel_entropy = np.trapz(integrand, x)

    return rel_entropy
