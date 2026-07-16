const f32 = Math.fround;

const LOG2_E = f32(1.4426950408889634);
const LOG2_HIGH = f32(-0.693145751953125);
const LOG2_LOW = f32(-1.428606765330187e-6);
const TANH_P0 = f32(0.0001980960224);
const TANH_P1 = f32(0.001394256484);
const TANH_P2 = f32(0.008333456703);
const TANH_P3 = f32(0.04166637361);
const TANH_P4 = f32(0.16666665941423425);
const TANH_LIMIT = f32(8.664339742);

export function createGemmaGeluLut(scale: number): Float32Array {
  const geluFactor = f32(0.7978845608028654);
  const cubicFactor = f32(0.044715);
  const gridScale = f32(scale);
  const lut = new Float32Array(256);
  for (let code = -128; code < 128; code += 1) {
    const value = f32(code * gridScale);
    const cube = f32(f32(value * value) * value);
    const tanhInput = f32(geluFactor * f32(value + f32(cubicFactor * cube)));
    const tanh = tanhF32(tanhInput);
    lut[code + 128] = f32(f32(0.5 * value) * f32(1 + tanh));
  }
  return lut;
}

function tanhF32(value: number): number {
  const magnitude = Math.abs(value);
  if (!(magnitude <= TANH_LIMIT)) return value < 0 ? -1 : 1;

  const exponent = roundTiesToEven(f32(magnitude * LOG2_E));
  let correction = f32(exponent * LOG2_HIGH);
  let reduced = f32(magnitude + correction);
  let offset = f32(reduced - magnitude);
  let residual = f32(
    f32(f32(magnitude - f32(reduced - offset)) + f32(correction - offset)) + 0,
  );

  correction = f32(exponent * LOG2_LOW);
  let next = f32(reduced + correction);
  offset = f32(next - reduced);
  residual = f32(
    f32(f32(reduced - f32(next - offset)) + f32(correction - offset)) + residual,
  );
  reduced = next;

  let polynomial = TANH_P0;
  polynomial = multiplyAdd(polynomial, reduced, TANH_P1);
  polynomial = multiplyAdd(polynomial, reduced, TANH_P2);
  polynomial = multiplyAdd(polynomial, reduced, TANH_P3);

  let product = f32(reduced * polynomial);
  let productError = multiplyAdd(
    residual,
    polynomial,
    multiplySubtract(reduced, polynomial, product),
  );
  next = f32(product + TANH_P4);
  offset = f32(next - product);
  productError = f32(
    f32(f32(product - f32(next - offset)) + f32(TANH_P4 - offset)) + productError,
  );
  product = next;

  let squaredProduct = f32(reduced * product);
  let squaredError = multiplyAdd(
    reduced,
    productError,
    multiplyAdd(residual, product, multiplySubtract(reduced, product, squaredProduct)),
  );
  next = f32(squaredProduct + 0.5);
  offset = f32(next - squaredProduct);
  productError = f32(
    f32(f32(squaredProduct - f32(next - offset)) + f32(0.5 - offset)) + squaredError,
  );
  product = next;

  const square = f32(reduced * reduced);
  const squareError = multiplyAdd(
    f32(reduced + reduced),
    residual,
    multiplySubtract(reduced, reduced, square),
  );
  squaredProduct = f32(square * product);
  squaredError = multiplyAdd(
    square,
    productError,
    multiplyAdd(squareError, product, multiplySubtract(square, product, squaredProduct)),
  );
  next = f32(reduced + squaredProduct);
  offset = f32(next - reduced);
  productError = f32(
    f32(f32(reduced - f32(next - offset)) + f32(squaredProduct - offset)) +
      f32(residual + squaredError),
  );
  product = next;

  next = f32(1 + product);
  productError = f32(f32(f32(1 - next) + product) + productError);
  product = scalePowerOfTwo(next, exponent);
  productError = scalePowerOfTwo(productError, exponent);

  const reciprocal = f32(1 / product);
  const reciprocalError = f32(
    reciprocal * subtractMultiply(productError, reciprocal, subtractMultiply(product, reciprocal, 1)),
  );
  const difference = f32(product - reciprocal);
  const differenceError = f32(
    f32(f32(f32(product - difference) - reciprocal) + productError) - reciprocalError,
  );
  const sum = f32(product + reciprocal);
  const sumError = f32(
    f32(f32(f32(product - sum) + reciprocal) + productError) + reciprocalError,
  );
  const inverseSum = f32(1 / sum);
  const quotient = f32(difference * inverseSum);
  const quotientError = multiplySubtract(inverseSum, difference, quotient);
  const inverseError = subtractMultiply(sumError, inverseSum, subtractMultiply(sum, inverseSum, 1));
  const correctionResult = multiplyAdd(
    quotient,
    inverseError,
    multiplyAdd(differenceError, inverseSum, quotientError),
  );
  const result = f32(quotient + correctionResult);
  return value < 0 ? -result : result;
}

function roundTiesToEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction > 0.5) return floor + 1;
  if (fraction < 0.5 || (floor & 1) === 0) return floor;
  return floor + 1;
}

function scalePowerOfTwo(value: number, exponent: number): number {
  const firstExponent = exponent >> 1;
  return f32(f32(value * 2 ** firstExponent) * 2 ** (exponent - firstExponent));
}

function multiplyAdd(left: number, right: number, addend: number): number {
  return f32(left * right + addend);
}

function multiplySubtract(left: number, right: number, subtrahend: number): number {
  return f32(left * right - subtrahend);
}

function subtractMultiply(left: number, right: number, minuend: number): number {
  return f32(minuend - left * right);
}