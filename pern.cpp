#include <bits/stdc++.h>
#include <chrono>
#include <fstream>

#define ll long long int
#define ld long double
#define DEGREE 2
#define ALPHA 10e-8
#define BETA 10e-5
#define GAMMA 10e-5

typedef struct PublicKey {
	std::vector<std::vector<ll>> F; // Coefficients of the Public Polynomial System
} PublicKey;

typedef struct PrivateKey {
	ll n; // Number of Variables
	ll l; // Defines Range of Input Message
	ll lg; // Defines Range of Polynomial Coefficients
	std::vector<std::vector<ll>> phi; // Polynomial System Phi
	ll mPhi; // Largest Value in Codomain of Phi
	std::vector<std::vector<ll>> psi; // Polynomial System Psi
	ll mPsi; // Largest Value in Codomain of Psi
	std::vector<ll> r; // r values r1,..., rn
	std::vector<std::vector<ll>> T; // Random Affine Transformation T
	ll q; // Prime q
} PrivateKey;

// Output in range [0, b - 1]
ll Modulo(ll a, ll b) {
	ll mod = a % b;
	mod = (mod + b) % b;
	return mod;
}

// Output in range (-b / 2, b / 2]
ll LeastAbsoluteRemainder(ll a, ll b) {
	ll lar = a % b;
	lar = (lar + b) % b;
	lar = lar > b / 2 ? lar - b : lar;
	return lar;
}

// Compute nCr
ll Combinations(ll n, ll r) {
	if(r > n - r) r = n - r;
	if(r == 0) return 1;
	ll c = 1;
	for(ll i = 1; i <= r; ++i) {
		c *= n - r + i;
        c /= i;
	}
	return c;
}

// Compute Number of Possible Monomials for a Polynomial with n Variables and d Degree
ll NumberOfMonomials(ll n, ll d) {
	// for(ll i = 1; i <= d; ++i) {
	// 	ll c = Combinations(n + i - 1, i);
	// 	num += c;
	// }
	ll num = Combinations(n + d, d); // The Above Code can be Simplified to this using Combinatorial Identities
	return num;
}

// Generate List of Random Integers in the range [lowLimit, upLimit]
std::vector<ll> GenerateRandom(ll n, ll lowLimit, ll upLimit) {
	std::vector<ll> list;
	ll range = upLimit - lowLimit + 1;
	for(ll i = 0; i < n; ++i) {
		ll random = (std::rand() % range) + lowLimit;
		list.push_back(random);
	}
	return list;
}

// Generate List of Random Real Numbers in the range [lowLimit, upLimit]
std::vector<ld> GenerateRandom(ll n, ld lowLimit, ld upLimit) {
	std::vector<ld> list;
	ll range = (ll) upLimit - (ll) lowLimit + 1;
	for(ll i = 0; i < n; ++i) {
		ld random = (std::rand() % range) + lowLimit + ((ld) std::rand()) / RAND_MAX;
		list.push_back(random);
	}
	return list;
}

// Generate Coefficients for n Polynomials in m Variables and d Degree in range (-r / 2, r / 2)
std::vector<std::vector<ll>> GenerateCoefficients(ll n, ll m, ll d, ll r) {
	std::vector<std::vector<ll>> coefficients;
	ll numMonomials = NumberOfMonomials(m, d);
	ll lowLimit = LeastAbsoluteRemainder(r / 2 + 1, r);
	ll upLimit = LeastAbsoluteRemainder(r / 2, r);
	for(ll i = 0; i < n; ++i) {
		coefficients.push_back(GenerateRandom(numMonomials, lowLimit, upLimit));
	}
	return coefficients;
}

// Compute the Result of a Single Polynomial on some input x (Integers)
ll ComputePolynomialOutput(std::vector<ll> coefficients, std::vector<ll> x, ll degree) {
	x.push_back(1); // Appending the Constant Term (Degree 0 Term) in the Input --> x = [x1, x2, ..., xn, 1]
	ll result = 0;
	// TODO: Re-implement for Generalized Degree
	// Current Implementation Works only for Degree 2 Polynomials
	for(ll i = 0, k = 0; i < x.size(); ++i) {
		for(ll j = i; j < x.size(); ++j, ++k) {
			result += x[i] * x[j] * coefficients[k];
		}
	}
	return result;
}

// Compute the Result of the Polynomial System on some input x (Integers)
std::vector<ll> ComputePolynomialSystemOutput(std::vector<std::vector<ll>> coefficients, std::vector<ll> x, ll degree) {
	std::vector<ll> result;
	for(ll i = 0; i < coefficients.size(); ++i) {
		ll out = ComputePolynomialOutput(coefficients[i], x, degree);
		result.push_back(out);
	}
	return result;
}

// Compute the Result of a Single Polynomial on some input x (Real Numbers)
ld ComputePolynomialOutput(std::vector<ll> coefficients, std::vector<ld> x, ll degree) {
	x.push_back(1.0); // Appending the Constant Term (Degree 0 Term) in the Input --> x = [x1, x2, ..., xn, 1.0]
	ld result = 0;
	// TODO: Re-implement for Generalized Degree
	// Current Implementation Works only for Degree 2 Polynomials
	for(ll i = 0, k = 0; i < x.size(); ++i) {
		for(ll j = i; j < x.size(); ++j, ++k) {
			result += x[i] * x[j] * coefficients[k];
		}
	}
	return result;
}

// Compute the Result of the Polynomial System on some input x (Real Numbers)
std::vector<ld> ComputePolynomialSystemOutput(std::vector<std::vector<ll>> coefficients, std::vector<ld> x, ll degree) {
	std::vector<ld> result;
	for(ll i = 0; i < coefficients.size(); ++i) {
		ld out = ComputePolynomialOutput(coefficients[i], x, degree);
		result.push_back(out);
	}
	return result;
}

// Return Max Possible Value in Codomain of Polynomial System
ll GetMaxInCodomain(std::vector<std::vector<ll>> coefficients, ll n, ll degree, ll l) {
	
	// Replacing the Coefficients with their Absolute Values
	for(ll i = 0; i < coefficients.size(); ++i) {
		for(ll j = 0; j < coefficients[i].size(); ++j) {
			coefficients[i][j] = std::abs(coefficients[i][j]);
		}
	}
	
	// Setting the Values of All the Variables to l / 2 to Maximize Output
	std::vector<ll> x;
	for(ll i = 0; i < n; ++i) {
		x.push_back(l / 2);
	}
	
	// Returning the Max Value from the Result of All the Polynomials in the System
	std::vector<ll> result = ComputePolynomialSystemOutput(coefficients, x, degree);
	ll max = 0;
	for(ll i = 0; i < result.size(); ++i) {
		max = std::max(max, result[i]);
	}
	return max;
}

// Checks if a Number is Prime
bool IsPrime(ll num) {
	ll sqnum = std::sqrt(num);
	for(ll i = 2; i <= sqnum; ++i) {
		if(num % i == 0) return false;
	}
	return true;
}

// Finds the Next Prime Greater Than base
ll GetNextPrime(ll base) {
	ll nextPrime = base + 1;
	if(nextPrime <= 2) nextPrime = 2;
	else if(nextPrime % 2 == 0) nextPrime++;
	while(!IsPrime(nextPrime)) {
		nextPrime += 2;
	}
	return nextPrime;
}

// Returns the r Values for the given Prime q. Else Returns Empty List
std::vector<ll> ComputeRValues(ll n, ll q, ll mPhi, ll mPsi) {
	
	std::vector<ll> rValues;
	ll base = 2 * mPhi;
	ll prevBase;
	
	for(ll i = 1; i <= n; ++i) {
		prevBase = base;
		for(ll r = base + 1; r < q; ++r) {
			bool flag = true;
			for(ll k = 1; k <= 2 * mPsi; ++k) {
				if(std::abs(LeastAbsoluteRemainder(r * k, q)) <= 2 * mPhi) {
					flag = false;
					break;
				}
			}
			if(flag) {
				rValues.push_back(r);
				base = r;
				break;
			}
		}
		// If No More r Values are Found, Break the Loop
		if(base == prevBase) {
			break; 
		}
	}

	// If n Distinct r Values are not Found, Empty the Vector
	if(rValues.size() != n) {
		rValues.clear();
	}
	return rValues;
}

// Returns the r Values and Prime q by Automatically Updating the Prime q
std::tuple<ll, std::vector<ll>> GetRValues(ll n, ll mPhi, ll mPsi) {
	ll base = 4 * mPhi * mPsi; // Prime q > 4 * mPhi * mPsi
	ll q; // Large Prime Number
	std::vector<ll> rValues; // List of the r Values;
	while(rValues.size() != n) {
		q = GetNextPrime(base);
		rValues.clear();
		rValues = ComputeRValues(n, q, mPhi, mPsi);
		base = q;
	}
	return std::make_tuple(q, rValues);
}

// Returns the Central Map G = (phi + r * psi) mod q
std::vector<std::vector<ll>> GetCentralMap(std::vector<std::vector<ll>> phiCoefficients, std::vector<std::vector<ll>> psiCoefficients, std::vector<ll> rValues, ll q, ll n) {
	std::vector<std::vector<ll>> centralMap;
	ll numMonomials = phiCoefficients[0].size(); // Both phi and psi have Same Number of Monomials
	for(ll i = 0; i < n; ++i) {
		std::vector<ll> polyi;
		for(ll j = 0; j < numMonomials; ++j) {
			ll coeff = phiCoefficients[i][j] + psiCoefficients[i][j] * rValues[i];
			coeff = Modulo(coeff, q);
			polyi.push_back(coeff);
		}
		centralMap.push_back(polyi);
	}
	return centralMap;
}

// Generate a Random Affine Transformation over the Field (Fq)^n
std::vector<std::vector<ll>> GetAffineTransformation(ll n, ll q) {
	std::vector<std::vector<ll>> affineT;
	for(ll i = 0; i < n; ++i) {
		affineT.push_back(GenerateRandom(n, 0, q - 1));
	}
	return affineT;
}

// Multiply 2 Matrices of Any Dimensions (Integers)
std::vector<std::vector<ll>> MatrixMultiplication(std::vector<std::vector<ll>> matA, std::vector<std::vector<ll>> matB) {

	std::vector<std::vector<ll>> productMat;
	if(matA[0].size() != matB.size()) {
		std::cout << "ERROR : Matrix dimensions do not match for multiplication.\n";
		return productMat;
	}

	for(ll i = 0; i < matA.size(); ++i) {
		std::vector<ll> product;
		ll tmpProd;
		for(ll c = 0; c < matB[0].size(); ++c) {
			tmpProd = 0;
			for(ll j = 0; j < matA[i].size(); ++j) {
				tmpProd += matA[i][j] * matB[j][c];
			}
			product.push_back(tmpProd);
		}
		productMat.push_back(product);
	}
	return productMat;
}

// Multiply 2 Matrices of Any Dimensions (Real Numbers)
std::vector<std::vector<ld>> MatrixMultiplication(std::vector<std::vector<ld>> matA, std::vector<std::vector<ld>> matB) {

	std::vector<std::vector<ld>> productMat;
	if(matA[0].size() != matB.size()) {
		std::cout << "ERROR : Matrix dimensions do not match for multiplication.\n";
		return productMat;
	}

	for(ll i = 0; i < matA.size(); ++i) {
		std::vector<ld> product;
		ll tmpProd;
		for(ll c = 0; c < matB[0].size(); ++c) {
			tmpProd = 0;
			for(ll j = 0; j < matA[i].size(); ++j) {
				tmpProd += matA[i][j] * matB[j][c];
			}
			product.push_back(tmpProd);
		}
		productMat.push_back(product);
	}
	return productMat;
}

// Computes the Final Central Map (the Public Key) F = T o G(x)
std::vector<std::vector<ll>> GetFinalPolynomialMap(std::vector<std::vector<ll>> affineT, std::vector<std::vector<ll>> centralMapG, ll q) {
	std::vector<std::vector<ll>> polynomialMapF;
	polynomialMapF = MatrixMultiplication(affineT, centralMapG);
	for(ll i = 0; i < polynomialMapF.size(); ++i) {
		for(ll j = 0; j < polynomialMapF[i].size(); ++j) {
			polynomialMapF[i][j] = Modulo(polynomialMapF[i][j], q);
		}
	}
	return polynomialMapF;
}

// Returns the GCD and Coefficients of GCD(a, b)
std::tuple<ll, ll, ll> ExtendedEuclideanGCD(ll a, ll b) {
	std::tuple<ll, ll, ll> coefficients;
	if(b == 0) {
		coefficients = std::make_tuple(a, 1, 0);
	} else {
		ll x; // Coefficient of a
		ll y; // Coefficient of b
		ll gcd; // GCD of a and b
		std::tie(gcd, x, y) = ExtendedEuclideanGCD(b, Modulo(a, b));
		coefficients = std::make_tuple(gcd, y, x - y * (a / b));
	}
	return coefficients;
}

// Computes the Positive Multiplicative Inverse of b over a
ll MultiplicativeInverse(ll a, ll b) {
	ll inverse = 0;
	std::tie(std::ignore, std::ignore, inverse) = ExtendedEuclideanGCD(a, b); // Only get the coefficient of b
	inverse = Modulo(inverse, a); // Convert negative coefficients to positive ones
	return inverse;
}

// Invert a Square Matrix in the Integer Field Fq
std::vector<std::vector<ll>> MatrixInverse(std::vector<std::vector<ll>> mat, ll q) {
	
	// Initializing the Augmented Matrix
	std::vector<std::vector<ll>> augmentedMat;
	for(ll i = 0; i < mat.size(); ++i) {
		std::vector<ll> row(2 * mat[i].size());
		for(ll j = 0; j < mat[i].size(); ++j) {
			row[j] = mat[i][j];
			row[j + mat[i].size()] = i == j ? 1 : 0;
		}
		augmentedMat.push_back(row);
	}

	// Running Gaussian Elimination Algorithm to obtain the Inverse
	for(ll i = 0; i < augmentedMat.size(); ++i) {

		ll inv = MultiplicativeInverse(q, augmentedMat[i][i]);
		for(ll j = 0; j < augmentedMat[i].size(); ++j) {
			augmentedMat[i][j] = Modulo(augmentedMat[i][j] * inv, q);
		}

		for(ll j = 0; j < augmentedMat.size(); ++j) {
			if(j == i) continue; // Skip the current row
			ll scale = Modulo(0 - augmentedMat[j][i], q);
			for(ll k = 0; k < augmentedMat[j].size(); ++k) {
				augmentedMat[j][k] = Modulo(augmentedMat[j][k] + scale * augmentedMat[i][k], q);
			}
		}
	}

	// Extracting the Inverse from the Augmented Matrix
	std::vector<std::vector<ll>> inverseMat;
	for(ll i = 0; i < augmentedMat.size(); ++i) {
		std::vector<ll> row;
		for(ll j = augmentedMat[i].size() / 2; j < augmentedMat[i].size(); ++j) {
			row.push_back(augmentedMat[i][j]);
		}
		inverseMat.push_back(row);
	}
	
	return inverseMat;
}

// Invert a Square Matrix (Real Numbers)
std::vector<std::vector<ld>> MatrixInverse(std::vector<std::vector<ld>> mat) {
	
	// Initializing the Augmented Matrix
	std::vector<std::vector<ld>> augmentedMat;
	for(ll i = 0; i < mat.size(); ++i) {
		std::vector<ld> row(2 * mat[i].size());
		for(ll j = 0; j < mat[i].size(); ++j) {
			row[j] = mat[i][j];
			row[j + mat[i].size()] = i == j ? 1.0 : 0.0;
		}
		augmentedMat.push_back(row);
	}

	// Running Gaussian Elimination Algorithm to obtain the Inverse
	for(ll i = 0; i < augmentedMat.size(); ++i) {

		ld inv = 1.0 / augmentedMat[i][i];
		for(ll j = 0; j < augmentedMat[i].size(); ++j) {
			augmentedMat[i][j] *= inv;
		}

		for(ll j = 0; j < augmentedMat.size(); ++j) {
			if(j == i) continue; // Skip the current row
			ld scale = 0 - augmentedMat[j][i];
			for(ll k = 0; k < augmentedMat[j].size(); ++k) {
				augmentedMat[j][k] += scale * augmentedMat[i][k];
			}
		}
	}

	// Extracting the Inverse from the Augmented Matrix
	std::vector<std::vector<ld>> inverseMat;
	for(ll i = 0; i < augmentedMat.size(); ++i) {
		std::vector<ld> row;
		for(ll j = augmentedMat[i].size() / 2; j < augmentedMat[i].size(); ++j) {
			row.push_back(augmentedMat[i][j]);
		}
		inverseMat.push_back(row);
	}
	
	return inverseMat;
}

// Convert a 1 x n Matrix into an n x 1 Vector
std::vector<std::vector<ll>> Vectorize(std::vector<ll> matA) {
	std::vector<std::vector<ll>> vecA;
	for(ll i = 0; i < matA.size(); ++i) {
		std::vector<ll> element = {matA[i]};
		vecA.push_back(element);
	}
	return vecA;
}

// Convert an n x 1 Vector into a 1 x n Matrix
std::vector<ll> DeVectorize(std::vector<std::vector<ll>> vecA) {
	std::vector<ll> matA;
	for(ll i = 0; i < vecA.size(); ++i) {
		matA.push_back(vecA[i][0]);
	}
	return matA;
}

// Compute the a Values (RHS of Phi(x)) and the b Values (RHS of Psi(x))
std::tuple<std::vector<ll>, std::vector<ll>> GetABValues(PrivateKey privateKey, std::vector<ll> inverseCipherText) {
	std::vector<ll> aValues;
	std::vector<ll> bValues;
	for(ll i = 0; i < privateKey.n; ++i) {
		for(ll b = 0; b <= privateKey.mPsi; ++b) {
			if(std::abs(LeastAbsoluteRemainder(inverseCipherText[i] - b * privateKey.r[i], privateKey.q)) < privateKey.mPhi) {
				bValues.push_back(b);
				aValues.push_back(LeastAbsoluteRemainder(inverseCipherText[i] - b * privateKey.r[i], privateKey.q));
				break;
			} else if(std::abs(LeastAbsoluteRemainder(inverseCipherText[i] + b * privateKey.r[i], privateKey.q)) < privateKey.mPhi) {
				bValues.push_back(0 - b);
				aValues.push_back(LeastAbsoluteRemainder(inverseCipherText[i] + b * privateKey.r[i], privateKey.q));
				break;
			}
		}
	}
	return std::make_tuple(aValues, bValues);
}

// Computes the Partial Derivatives of a Single Polynomial over All Variables
std::vector<ld> ComputePartialDerivatives(std::vector<ll> coefficients, std::vector<ld> x, ll degree) {
	// Only works for Quadratic Polynomial Equations
	std::vector<ld> xCopy = x;
	xCopy.push_back(1.0); // Pushing the Constant Term

	std::vector<ld> partialDerivatives;
	for(ll i = 0; i < x.size(); ++i) { // Only Compute the Partial Derivatives w.r.t. the Variables and not the Constant Term
		ld result = 0.0;
		for(ll j = 0, l = 0; j < xCopy.size(); ++j) {
			for(ll k = 0; k < xCopy.size(); ++k, ++l) {
				if(j == k) {
					if(i == j) {
						result += 2.0 * coefficients[l] * xCopy[j]; // derivative of xj^2 w.r.t. xj
					} else {
						result += 0.0; // derivative of xj^2 w.r.t. xi
					}
				} else {
					if(i == j) {
						result += coefficients[l] * xCopy[k]; // derivative of xj.xk w.r.t. xj
					} else if(i == k) {
						result += coefficients[l] * xCopy[j]; // derivative of xj.xk w.r.t. xk
					} else {
						result += 0.0; // derivative of xj.xk w.r.t. xi
					}
				}
			}
		}
		partialDerivatives.push_back(result);
	}

	return partialDerivatives;
}

// Computes the Jacobian of the Polynomial System
std::vector<std::vector<ld>> ComputeJacobianMatrix(std::vector<std::vector<ll>> function, std::vector<ld> x, ll degree) {
	std::vector<std::vector<ld>> jacobian;
	for(ll i = 0; i < function.size(); ++i) {
		jacobian.push_back(ComputePartialDerivatives(function[i], x, degree));
	}
	return jacobian;
}

// Returns the Transpose of a Matrix
std::vector<std::vector<ld>> MatrixTranspose(std::vector<std::vector<ld>> mat) {
	std::vector<std::vector<ld>> transpose;
	for(ll i = 0; i < mat[0].size(); ++i) {
		std::vector<ld> row;
		for(ll j = 0; j < mat.size(); ++j) {
			row.push_back(mat[j][i]);
		}
		transpose.push_back(row);
	}
	return transpose;
}

// Compute Norm of a Vector = 0.5 * (vec[1]^2 + vec[2]^2 + ... + vec[n]^2)
ld ComputeNorm(std::vector<ld> vec) {
	ld norm = 0.0;
	for(ll i = 0; i < vec.size(); ++i) {
		norm += vec[i] * vec[i];
	}
	norm *= 0.5;
	return norm;
}

// Returns the Max Absolute Component of a Vector
ld MaxAbsoluteComponent(std::vector<ld> vec) {
	ld max = 0.0;
	for(ll i = 0; i < vec.size(); ++i) {
		max = std::abs(vec[i]) > max ? std::abs(vec[i]) : max;
	}
	return max;
}

// Solve the Non Linear Polynomial Equation System using the Levenberg-Marquardt Method
std::vector<ll> SolveNonLinearEquationSystem(PrivateKey privateKey, std::vector<ll> aValues, std::vector<ll> bValues, ll degree) {
	
	ld alpha = ALPHA; // Search Parameter
	ld beta = BETA; // Search Parameter
	ld gamma = GAMMA; // Search Parameter

	std::vector<std::vector<ll>> h; // H(x) = Concatenation of Phi(x) - aValues and Psi(x) - bValues
	for(ll i = 0; i < privateKey.phi.size(); ++i) {
		h.push_back(privateKey.phi[i]);
		h[h.size() - 1][privateKey.phi[i].size() - 1] -= aValues[i]; // Phi(x) - aValues
	}
	for(ll i = 0; i < privateKey.psi.size(); ++i) {
		h.push_back(privateKey.psi[i]);
		h[h.size() - 1][privateKey.psi[i].size() - 1] -= bValues[i]; // Psi(x) - bValues
	}

	std::vector<ll> decryptedMessage;

	bool flag = false;
	ld lowLimit = - privateKey.l / 2.0;
	ld upLimit = privateKey.l / 2.0;
	std::vector<ld> x0;
	do {
		x0.clear();
		x0 = GenerateRandom(privateKey.n, lowLimit, upLimit); // Initial Random Guess in Range [-l / 2, l / 2]
		std::vector<std::vector<ld>> hResult = {ComputePolynomialSystemOutput(h, x0, degree)}; // Computing h(x0) : (1 x 2n)
		std::vector<std::vector<ld>> hJacobian = ComputeJacobianMatrix(h, x0, degree); // Computing Jacobian of h(x) at x0 : (2n x n)
		std::vector<std::vector<ld>> negativeIdentityMatrix = {{-1.0}}; // Negative Identity Matrix : (1 x 1)

		std::vector<std::vector<ld>> e = MatrixMultiplication(MatrixMultiplication(negativeIdentityMatrix, hResult), hJacobian); // e = -h(x0) * Jh(x0) : (1 x n)
		std::vector<std::vector<ld>> s = MatrixMultiplication(MatrixTranspose(hJacobian), hJacobian); // s = Jh(x0)^T * Jh(x0) : (n x n)
		std::vector<std::vector<ld>> d0 = MatrixMultiplication(e, MatrixInverse(s)); // d0 * s = e : (1 x n)
		std::vector<std::vector<ld>> ed0Product = MatrixMultiplication(e, MatrixTranspose(d0)); // e * d0^T : (1 x 1)

		ld t0 = 0.0; // Step Size
		std::vector<ld> x1; // x1 = x0 + t0 * d0
		std::vector<std::vector<ld>> hResultNew; // h(x1)
		ld thetaOldX = ComputeNorm(hResult[0]); // theta(x0) = Norm of h(x0)
		ld thetaNewX = 0.0; // theta(x1)
		for(ll i = 0; i >= 0; ++i) {
			ld stepSize = std::pow(beta, i);
			x1.clear();
			for(ll j = 0; j < x0.size(); ++j) {
				x1.push_back(x0[j] + stepSize * d0[0][j]);
			}
			
			hResultNew.clear();
			hResultNew = {ComputePolynomialSystemOutput(h, x1, degree)};
			thetaNewX = ComputeNorm(hResultNew[0]);
			ld delta = (-alpha) * stepSize * ed0Product[0][0];
			if(thetaNewX - thetaOldX <= delta) {
				t0 = stepSize;
			} else {
				continue;
			}
			
			std::vector<ld> deltaX; // x1 = x0 + deltaX
			for(ll i = 0; i < x0.size(); ++i) {
				deltaX.push_back(x1[i] - x0[i]);
			}
			ld maxComponent = MaxAbsoluteComponent(deltaX);
			if(maxComponent < gamma) {
				break;
			}
		}

		decryptedMessage.clear();
		for(ll i = 0; i < x1.size(); ++i) {
			decryptedMessage.push_back((ll) std::round(x1[i]));
		}

		std::vector<ll> hVerify = ComputePolynomialSystemOutput(h, decryptedMessage, degree);
		flag = true;
		for(ll i = 0; i < hVerify.size(); ++i) {
			if(hVerify[i] != 0) {
				flag = false;
				break;
			}
		}
	} while(!flag);

	return decryptedMessage;
}

// Write Public Key to File
void WritePublicKeyToFile(PublicKey publicKey, std::string filename) {

	std::ofstream publicKeyFile ("PublicKey.txt");
	
	publicKeyFile << "F:\n";
	for(ll i = 0; i < publicKey.F.size(); ++i) {
		for(ll j = 0; j < publicKey.F[i].size(); ++j) {
			publicKeyFile << publicKey.F[i][j] << " ";
		}
		publicKeyFile << "\n";
	}
	
	return;
}

// Write Private Key to File
void WritePrivateKeyToFile(PrivateKey privateKey, std::string filename) {
	
	std::ofstream privateKeyFile (filename);

	privateKeyFile << "n:\n" << privateKey.n << "\n\n";

	privateKeyFile << "l:\n" << privateKey.l << "\n\n";

	privateKeyFile << "lg:\n" << privateKey.lg << "\n\n";
	
	privateKeyFile << "Phi:\n";
	for(ll i = 0; i < privateKey.phi.size(); ++i) {
		for(ll j = 0; j < privateKey.phi[i].size(); ++j) {
			privateKeyFile << privateKey.phi[i][j] << " ";
		}
		privateKeyFile << "\n";
	}

	privateKeyFile << "\nMPhi:\n" << privateKey.mPhi << "\n\n";
	
	privateKeyFile << "Psi:\n";
	for(ll i = 0; i < privateKey.psi.size(); ++i) {
		for(ll j = 0; j < privateKey.psi[i].size(); ++j) {
			privateKeyFile << privateKey.psi[i][j] << " ";
		}
		privateKeyFile << "\n";
	}
	
	privateKeyFile << "\nMPsi:\n" << privateKey.mPsi << "\n\n";
	
	privateKeyFile << "r Values:\n";
	for(ll i = 0; i < privateKey.r.size(); ++i) {
		privateKeyFile << privateKey.r[i] << " ";
	}
	privateKeyFile << "\n\n";
	
	privateKeyFile << "T:\n";
	for(ll i = 0; i < privateKey.T.size(); ++i) {
		for(ll j = 0; j < privateKey.T[i].size(); ++j) {
			privateKeyFile << privateKey.T[i][j] << " ";
		}
		privateKeyFile << "\n";
	}
	
	privateKeyFile << "\nq:\n" << privateKey.q << "\n";
	
	return;
}

// Generate Public Private Key Pair
std::pair<PublicKey, PrivateKey> GenerateKeyPair(ll n, ll l, ll lg, ll degree) {
	
	std::vector<std::vector<ll>> phiCoefficients = GenerateCoefficients(n, n, degree, lg); // Coefficients for the Phi Polynomial System
	std::vector<std::vector<ll>> psiCoefficients = GenerateCoefficients(n, n, degree, lg); // Coefficients for the Psi Polynomial System
	
	ll mPhi = GetMaxInCodomain(phiCoefficients, n, degree, l); // Largest Value in Codomain of Phi
	ll mPsi = GetMaxInCodomain(psiCoefficients, n, degree, l); // Largest Value in Codomain of Psi
	
	ll q; // Large Prime q
	std::vector<ll> rValues; // r Values
	std::tie(q, rValues) = GetRValues(n, mPhi, mPsi); // r Values and Prime q

	std::vector<std::vector<ll>> centralMapG = GetCentralMap(phiCoefficients, psiCoefficients, rValues, q, n); // The Central Map G
	
	std::vector<std::vector<ll>> affineT = GetAffineTransformation(n, q); // Random Affine Transformation T

	std::vector<std::vector<ll>> polynomialMapF = GetFinalPolynomialMap(affineT, centralMapG, q); // Final Polynomial Map and Public Key F

	// Save to publicKey
	std::cout << "Saving Public Key...\n";
	PublicKey publicKey;
	publicKey.F = polynomialMapF;
	WritePublicKeyToFile(publicKey, "PublicKey.txt");

	// Save to privateKey
	std::cout << "Saving Private Key...\n";
	PrivateKey privateKey;
	privateKey.n = n;
	privateKey.l = l;
	privateKey.lg = lg;
	privateKey.phi = phiCoefficients;
	privateKey.mPhi = mPhi;
	privateKey.psi = psiCoefficients;
	privateKey.mPsi = mPsi;
	privateKey.r = rValues;
	privateKey.T = affineT;
	privateKey.q = q;
	WritePrivateKeyToFile(privateKey, "PrivateKey.txt");

	return std::make_pair(publicKey, privateKey);
}

// Encrypting the Input Message Using the Public Key
std::vector<ll> EncryptMessage(PublicKey publicKey, std::vector<ll> message, ll degree) {
	std::vector<ll> cipherText = ComputePolynomialSystemOutput(publicKey.F, message, degree);
	return cipherText;
}

// Decrypting the Cipher Text Using the Private Key
std::vector<ll> DecryptCipherText(PrivateKey privateKey, std::vector<ll> cipherText, ll degree) {
	// Brings the Cipher Text to the Finite Field Fq
	for(ll i = 0; i < cipherText.size(); ++i) {
		cipherText[i] = Modulo(cipherText[i], privateKey.q);
	}

	// Computing T^-1(c)
	std::vector<std::vector<ll>> inverseT = MatrixInverse(privateKey.T, privateKey.q);
	std::vector<ll> inverseCipherText = DeVectorize(MatrixMultiplication(inverseT, Vectorize(cipherText))); // ComputePolynomialSystemOutput() not used as T is a system of linear polynomials
	
	// Computing the RHS of Phi(x) and Psi(x)
	std::vector<ll> aValues;
	std::vector<ll> bValues;
	std::tie(aValues, bValues) = GetABValues(privateKey, inverseCipherText); // Get the RHS of Phi(x) and Psi(x)

	// Solving the Non Linear Polynomial System using Levenberg-Marquardt Method
	std::vector<ll> decryptedMessage = SolveNonLinearEquationSystem(privateKey, aValues, bValues, degree);

	return decryptedMessage;
}

int main() {

	std::srand(std::time(0)); // Setting a New Seed Value on Every Run
	
	std::cout << "Enter Security Parameters (n L Lg degree):\n";
	ll n; // Number of Variables in System of Polynomial Equations
	ll l; // Odd Positive Integer
	ll lg; // Odd Positive Integer
	ll degree = DEGREE; // Maximum Degree of the Monomials in the Polynomial System
	std::cin >> n >> l >> lg;

	/**
	 * ################################## Generating Public Private Key Pair ##################################
	 */

	std::cout << "\nGenerating Key Pair...\n";
	auto startTimeKG = std::chrono::high_resolution_clock::now();
	
	std::pair<PublicKey, PrivateKey> keyPair = GenerateKeyPair(n, l, lg, degree);
	PublicKey publicKey = keyPair.first;
	PrivateKey privateKey = keyPair.second;
	
	auto stopTimeKG = std::chrono::high_resolution_clock::now();
	auto durationKG = std::chrono::duration_cast<std::chrono::milliseconds>(stopTimeKG - startTimeKG);
	double execTime = durationKG.count() / 1000.0;
	std::cout << "Time taken to generate Key Pair = " << execTime << "s\n\n";

	/**
	 * ################################## Encrypting the Message using the Public Key ##################################
	 */

	std::cout << "Enter Message to Encrypt (Enter " << n << " numbers in the range [" << std::floor(-l / 2.0) + 1 << ", " << std::floor(l / 2.0) << "]):\n";
	std::vector<ll> message(n); // Input to Encrypt
	for(ll i = 0; i < n; ++i) {
		std::cin >> message[i];
	}

	std::cout << "\nEncrypting Input Message...\n";
	auto startTimeEnc = std::chrono::high_resolution_clock::now();
	
	std::vector<ll> cipherText = EncryptMessage(publicKey, message, degree);

	auto stopTimeEnc = std::chrono::high_resolution_clock::now();
	auto durationEnc = std::chrono::duration_cast<std::chrono::nanoseconds>(stopTimeEnc - startTimeEnc);
	execTime = durationEnc.count() / 1000.0;

	std::cout << "Encrypted Cipher Text: ";
	for(ll i = 0; i < cipherText.size(); ++i) {
		std::cout << cipherText[i] << " ";
	}
	std::cout << "\n";
	std::cout << "Time taken to Encrypt Message = " << execTime << "us\n\n";

	/**
	 * ################################## Decrypting the Cipher Text using the Private Key ##################################
	 */

	std::cout << "Decrypting Cipher Text...\n";
	auto startTimeDec = std::chrono::high_resolution_clock::now();
	
	std::vector<ll> decryptedMessage = DecryptCipherText(privateKey, cipherText, degree);

	auto stopTimeDec = std::chrono::high_resolution_clock::now();
	auto durationDec = std::chrono::duration_cast<std::chrono::milliseconds>(stopTimeDec - startTimeDec);
	execTime = durationDec.count() / 1000.0;

	std::cout << "Decrypted Message: ";
	for(ll i = 0; i < decryptedMessage.size(); ++i) {
		std::cout << decryptedMessage[i] << " ";
	}
	std::cout << "\n";
	std::cout << "Time taken to Decrypt Message = " << execTime << "s\n";

	return 0;
}