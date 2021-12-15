#include <bits/stdc++.h>

#define ll long long int
#define DEGREE 2

typedef struct PublicKey {
	std::vector<std::vector<ll>> F;
} PublicKey;

typedef struct PrivateKey {
	std::vector<std::vector<ll>> phi;
	std::vector<std::vector<ll>> psi;
	std::vector<ll> r;
	std::vector<std::vector<ll>> T;
} PrivateKey;

PublicKey publicKey; // Public Key
PrivateKey privateKey; // Private Key

// std::srand((unsigned) std::time(NULL)); // Setting a New Seed Value on Every Run

// Output in range [0, b - 1]
ll Modulo(ll a, ll b) {
	ll mod = a % b;
	mod = (mod + b) % b;
	return mod;
}

// Output in range (-b / 2, b / 2]
ll LeastAbsoluteRemainder(ll a, ll b) {
	ll lar = a % b;
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

// Generate List of Random Numbers in the range [lowLimit, upLimit]
std::vector<ll> GenerateRandom(ll n, ll lowLimit, ll upLimit) {
	std::vector<ll> list;
	ll range = upLimit - lowLimit + 1;
	for(ll i = 0; i < n; ++i) {
		ll random = (std::rand() % range) + lowLimit;
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

// Compute the Result of a Single Polynomial on some input x
ll ComputePolynomialOutput(std::vector<ll> coefficients, std::vector<ll> x, ll degree) {
	x.push_back(1); // Appending the Constant Term (Degree 0 Term) in the Input --> x = [x1, x2, ..., xn, 1]
	ll numMonomials = NumberOfMonomials((ll) x.size(), degree);
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

// Compute the Result of the Polynomial System on some input x
std::vector<ll> ComputePolynomialSystemOutput(std::vector<std::vector<ll>> coefficients, std::vector<ll> x, ll degree) {
	std::vector<ll> result;
	for(ll i = 0; i < coefficients.size(); ++i) {
		ll out = ComputePolynomialOutput(coefficients[i], x, degree);
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
std::pair<ll, std::vector<ll>> GetRValues(ll n, ll mPhi, ll mPsi) {
	ll base = 4 * mPhi * mPsi; // Prime q > 4 * mPhi * mPsi
	ll q; // Large Prime Number
	std::vector<ll> rValues; // List of the r Values;
	while(rValues.size() != n) {
		q = GetNextPrime(base);
		rValues.clear();
		rValues = ComputeRValues(n, q, mPhi, mPsi);
		base = q;
	}
	return std::make_pair(q, rValues);
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

// Generate a Random Affine Tranformation over the Field (Fq)^n
std::vector<std::vector<ll>> GetAffineTransformation(ll n, ll q) {
	std::vector<std::vector<ll>> affineT;
	// TODO
	return affineT;
}

// Computes the Final Central Map (the Public Key) F = T o G(x)
std::vector<std::vector<ll>> GetFinalPolynomialMap(std::vector<std::vector<ll>> affineT, std::vector<std::vector<ll>> centralMapG) {
	std::vector<std::vector<ll>> polynomialMapF;
	// TODO
	return polynomialMapF;
}

// Generate Public Private Key Pair
void GenerateKeyPair(ll n, ll l, ll lg, ll degree) {
	
	std::vector<std::vector<ll>> phiCoefficients = GenerateCoefficients(n, n, degree, lg); // Coefficients for the Phi Polynomial System
	std::vector<std::vector<ll>> psiCoefficients = GenerateCoefficients(n, n, degree, lg); // Coefficients for the Psi Polynomial System
	
	ll mPhi = GetMaxInCodomain(phiCoefficients, n, degree, l); // Largest Value in Codomain of Phi
	ll mPsi = GetMaxInCodomain(psiCoefficients, n, degree, l); // Largest Value in Codomain of Psi
	
	std::pair<ll, std::vector<ll>> result = GetRValues(n, mPhi, mPsi); // r Values and Prime q
	ll q = result.first; // Large Prime q
	std::vector<ll> rValues = result.second; // r Values

	std::vector<std::vector<ll>> centralMapG = GetCentralMap(phiCoefficients, psiCoefficients, rValues, q, n); // The Central Map G
	
	std::vector<std::vector<ll>> affineT = GetAffineTransformation(n, q); // Random Affine Transformation T

	std::vector<std::vector<ll>> polynomialMapF = GetFinalPolynomialMap(affineT, centralMapG); // Final Polynomial Map and Public Key F

	// Save to publicKey
	publicKey.F = polynomialMapF;

	// Save to privateKey
	privateKey.phi = phiCoefficients;
	privateKey.psi = psiCoefficients;
	privateKey.r = rValues;
	privateKey.T = affineT;

	return;
}

int main() {
	
	std::cout << "Enter Security Parameters (n L Lg degree):\n";
	ll n; // Number of Variables in System of Polynomial Equations
	ll l; // Odd Positive Integer
	ll lg; // Odd Positive Integer
	ll degree = DEGREE; // Maximum Degree of the Monomials in the Polynomial System
	std::cin >> n >> l >> lg;

	GenerateKeyPair(n, l, lg, degree);

	std::cout << "Enter Input to Encrypt (Enter " << n << " numbers in the range [" << std::floor(-l / 2.0) + 1 << ", " << std::floor(l / 2.0) << "]):\n";
	ll msg[n]; // Input to Encrypt
	for(ll i = 0; i < n; ++i) {
		std::cin >> msg[i];
	}

	// TODO
	// std::cout << NumberOfMonomials(n, 2) << "\n";

	return 0;
}