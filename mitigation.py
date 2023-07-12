import numpy as np
from scipy import linalg
import collections.abc

def get_probabilities(counts, shots=1024, in_range = (0.01, 0.99)) :
  """
  Get mitigated probabilities from a JobResult count dict using the contrast filter with default 1% intensity
  Args: counts dictionary of counts {k1:v1, k2:v2,...} or list of dictionaries [{k1:v1,..}{k1:v1,...}]
  """
  if (isinstance(counts, collections.abc.Sequence) ) :
    new_counts = []
    for  dist in counts :
      new_counts.append(_mitigate_contrast1d(dist, shots, in_range))
    return new_counts
  else :
    return _mitigate_contrast1d(counts, shots, in_range)

def get_counts(counts, shots=1024, in_range = (0.01, 0.99)) :
  """
  Get mitigated counts
  """
  probs = get_probabilities(counts, shots, in_range)
  if (isinstance(probs, collections.abc.Sequence) ) :
    new_counts = []
    for  dist in probs :
      new_counts.append({key:int(round((prob * shots))) for key, prob in dist.items()})
    return new_counts
  else :
    return {key:int(round((prob * shots))) for key, prob in probs.items()}
    
# convert a list if probabilities to an image
def _prepare_image(probs):
  dim     = len(probs)
  pixels  = [val for key,val in probs.items()]
  img     = np.ndarray(shape=(dim,1), dtype=float)
  for col in range(dim):
      img[col] = pixels[col]  
  return img
  
# 1d array mitigator helper
def _mitigate_contrast1d(counts, shots=1024, in_range = (0.15, 0.4)) :
  probs = {key:count/shots for key,count in counts.items()}
  img   = _prepare_image(probs)
  
  # apply filter
  img_contrast = rescale_intensity(img, in_range=in_range) # exposure.
  
  # normalize
  array_1d      = img_contrast.flatten()
  norm          = linalg.norm(array_1d, ord=1)
  
  array_1d_norm = array_1d / norm if norm > 0 else array_1d

  # Build new probs list
  new_probs = {}
  i = 0
  for key in probs.keys():
    new_probs[key] = array_1d_norm[i]
    i += 1
  return new_probs

######################### Expectation values  (slow)
 
def expval (probs, pauli_list):
  """
  Calculate expectation values
  Args: probs: array of dicts [{},...],  pauli_list: Measurement strings ['ZZZ',...]
  Return: [e1, e2,...]
  """
  if ( not (isinstance(probs, collections.abc.Sequence) ) ) :
    raise Exception("Probabilities must be an array of dictionary counts {s1: count1, s2: count2,...}")
    
  i = 0;
  vals = []
  for dist in probs :
    counts = Counts(dist)
    paulis = PauliList(pauli_list[i])
    expvals, variances = pauli_expval_with_variance(counts, paulis)
    vals.append(expvals[0])
    i += 1
  return vals
    
from qiskit.quantum_info import Pauli, PauliList
from qiskit.result import Counts

def _parity(integer: int) -> int:
    """Return the parity of an integer"""
    return bin(integer).count("1") % 2

def _paulis2inds(paulis: PauliList) : # -> list[int]:
    """Convert PauliList to diagonal integers.
    These are integer representations of the binary string with a
    1 where there are Paulis, and 0 where there are identities.
    """
    # Treat Z, X, Y the same
    nonid = paulis.z | paulis.x

    inds = [0] * paulis.size
    # bits are packed into uint8 in little endian
    # e.g., i-th bit corresponds to coefficient 2^i
    packed_vals = np.packbits(nonid, axis=1, bitorder="little")
    for i, vals in enumerate(packed_vals):
        for j, val in enumerate(vals):
            inds[i] += val.item() * (1 << (8 * j))
    return inds

# https://qiskit.org/documentation/stubs/qiskit.result.Counts.html
def pauli_expval_with_variance(counts: Counts, paulis: PauliList) : # -> tuple[np.ndarray, np.ndarray]:
    """Return array of expval and variance pairs for input Paulis.
    Note: All non-identity Pauli's are treated as Z-paulis, assuming
    that basis rotations have been applied to convert them to the
    diagonal basis.
    """
    # Diag indices
    size = len(paulis)
    diag_inds = _paulis2inds(paulis)

    expvals = np.zeros(size, dtype=float)
    denom = 0  # Total shots for counts dict
    for bin_outcome, freq in counts.items():
        outcome = int(bin_outcome, 2)
        denom += freq
        for k in range(size):
            coeff = (-1) ** _parity(diag_inds[k] & outcome)
            expvals[k] += freq * coeff

    # Divide by total shots
    if denom > 0 :
      expvals /= denom 

    # Compute variance
    variances = 1 - expvals**2
    return expvals, variances

################################# Image stuff
dtype_range = {bool: (False, True),
               np.bool_: (False, True),
               float: (-1, 1),
               np.float_: (-1, 1),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
  
DTYPE_RANGE = dtype_range.copy()
DTYPE_RANGE.update((d.__name__, limits) for d, limits in dtype_range.items())
DTYPE_RANGE.update({'uint10': (0, 2 ** 10 - 1),
                    'uint12': (0, 2 ** 12 - 1),
                    'uint14': (0, 2 ** 14 - 1),
                    'bool': dtype_range[bool],
                    'float': dtype_range[np.float64]})

def rescale_intensity(image, in_range='image', out_range='dtype'):
  """Return image after stretching or shrinking its intensity levels.

  The desired intensity range of the input and output, `in_range` and
  `out_range` respectively, are used to stretch or shrink the intensity range
  of the input image. See examples below.

  Parameters
  ----------
  image : array
      Image array.
  in_range, out_range : str or 2-tuple, optional
      Min and max intensity values of input and output image.
      The possible values for this parameter are enumerated below.

      'image'
          Use image min/max as the intensity range.
      'dtype'
          Use min/max of the image's dtype as the intensity range.
      dtype-name
          Use intensity range based on desired `dtype`. Must be valid key
          in `DTYPE_RANGE`.
      2-tuple
          Use `range_values` as explicit min/max intensities.

  Returns
  -------
  out : array
      Image array after rescaling its intensity. This image is the same dtype
      as the input image.

  Notes
  -----
  .. versionchanged:: 0.17
      The dtype of the output array has changed to match the input dtype, or
      float if the output range is specified by a pair of values.

  See Also
  --------
  equalize_hist

  Examples
  --------
  By default, the min/max intensities of the input image are stretched to
  the limits allowed by the image's dtype, since `in_range` defaults to
  'image' and `out_range` defaults to 'dtype':

  >>> image = np.array([51, 102, 153], dtype=np.uint8)
  >>> rescale_intensity(image)
  array([  0, 127, 255], dtype=uint8)

  It's easy to accidentally convert an image dtype from uint8 to float:

  >>> 1.0 * image
  array([ 51., 102., 153.])

  Use `rescale_intensity` to rescale to the proper range for float dtypes:

  >>> image_float = 1.0 * image
  >>> rescale_intensity(image_float)
  array([0. , 0.5, 1. ])

  To maintain the low contrast of the original, use the `in_range` parameter:

  >>> rescale_intensity(image_float, in_range=(0, 255))
  array([0.2, 0.4, 0.6])

  If the min/max value of `in_range` is more/less than the min/max image
  intensity, then the intensity levels are clipped:

  >>> rescale_intensity(image_float, in_range=(0, 102))
  array([0.5, 1. , 1. ])

  If you have an image with signed integers but want to rescale the image to
  just the positive range, use the `out_range` parameter. In that case, the
  output dtype will be float:

  >>> image = np.array([-10, 0, 10], dtype=np.int8)
  >>> rescale_intensity(image, out_range=(0, 127))
  array([  0. ,  63.5, 127. ])

  To get the desired range with a specific dtype, use ``.astype()``:

  >>> rescale_intensity(image, out_range=(0, 127)).astype(np.int8)
  array([  0,  63, 127], dtype=int8)

  If the input image is constant, the output will be clipped directly to the
  output range:
  >>> image = np.array([130, 130, 130], dtype=np.int32)
  >>> rescale_intensity(image, out_range=(0, 127)).astype(np.int32)
  array([127, 127, 127], dtype=int32)
  """
  if out_range in ['dtype', 'image']:
      out_dtype = _output_dtype(image.dtype.type, image.dtype)
  else:
      out_dtype = _output_dtype(out_range, image.dtype)

  imin, imax = map(float, intensity_range(image, in_range))
  omin, omax = map(float, intensity_range(image, out_range,
                                          clip_negative=(imin >= 0)))

  if np.any(np.isnan([imin, imax, omin, omax])):
      print(
          "One or more intensity levels are NaN. Rescaling will broadcast "
          "NaN to the full image. Provide intensity levels yourself to "
          "avoid this. E.g. with np.nanmin(image), np.nanmax(image)."
          
      ) # utils.warn

  image = np.clip(image, imin, imax)

  if imin != imax:
      image = (image - imin) / (imax - imin)
      return np.asarray(image * (omax - omin) + omin, dtype=out_dtype)
  else:
      return np.clip(image, omin, omax).astype(out_dtype)

def _output_dtype(dtype_or_range, image_dtype):
    """Determine the output dtype for rescale_intensity.

    The dtype is determined according to the following rules:
    - if ``dtype_or_range`` is a dtype, that is the output dtype.
    - if ``dtype_or_range`` is a dtype string, that is the dtype used, unless
      it is not a NumPy data type (e.g. 'uint12' for 12-bit unsigned integers),
      in which case the data type that can contain it will be used
      (e.g. uint16 in this case).
    - if ``dtype_or_range`` is a pair of values, the output data type will be
      ``_supported_float_type(image_dtype)``. This preserves float32 output for
      float32 inputs.

    Parameters
    ----------
    dtype_or_range : type, string, or 2-tuple of int/float
        The desired range for the output, expressed as either a NumPy dtype or
        as a (min, max) pair of numbers.
    image_dtype : np.dtype
        The input image dtype.

    Returns
    -------
    out_dtype : type
        The data type appropriate for the desired output.
    """
    if type(dtype_or_range) in [list, tuple, np.ndarray]:
        # pair of values: always return float.
        return utils._supported_float_type(image_dtype)
    if type(dtype_or_range) == type:
        # already a type: return it
        return dtype_or_range
    if dtype_or_range in DTYPE_RANGE:
        # string key in DTYPE_RANGE dictionary
        try:
            # if it's a canonical numpy dtype, convert
            return np.dtype(dtype_or_range).type
        except TypeError:  # uint10, uint12, uint14
            # otherwise, return uint16
            return np.uint16
    else:
        raise ValueError(
            'Incorrect value for out_range, should be a valid image data '
            f'type or a pair of values, got {dtype_or_range}.'
        )

def intensity_range(image, range_values='image', clip_negative=False):
    """Return image intensity range (min, max) based on desired value type.

    Parameters
    ----------
    image : array
        Input image.
    range_values : str or 2-tuple, optional
        The image intensity range is configured by this parameter.
        The possible values for this parameter are enumerated below.

        'image'
            Return image min/max as the range.
        'dtype'
            Return min/max of the image's dtype as the range.
        dtype-name
            Return intensity range based on desired `dtype`. Must be valid key
            in `DTYPE_RANGE`. Note: `image` is ignored for this range type.
        2-tuple
            Return `range_values` as min/max intensities. Note that there's no
            reason to use this function if you just want to specify the
            intensity range explicitly. This option is included for functions
            that use `intensity_range` to support all desired range types.

    clip_negative : bool, optional
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.
    """
    if range_values == 'dtype':
        range_values = image.dtype.type

    if range_values == 'image':
        i_min = np.min(image)
        i_max = np.max(image)
    elif range_values in DTYPE_RANGE:
        i_min, i_max = DTYPE_RANGE[range_values]
        if clip_negative:
            i_min = 0
    else:
        i_min, i_max = range_values
    return i_min, i_max
      