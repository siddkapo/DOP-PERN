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
	// Affine Transform T
	// TODO
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
	// TODO
	ll result = 0;
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
ll MaxInCodomain(std::vector<std::vector<ll>> coefficients, ll n, ll degree, ll l) {
	
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
// Generate Public Private Key Pair
void GenerateKeyPair(ll n, ll l, ll lg, ll degree) {
	
	std::vector<std::vector<ll>> phiCoefficients = GenerateCoefficients(n, n, degree, lg); // Coefficients for the Phi Polynomial System
	std::vector<std::vector<ll>> psiCoefficients = GenerateCoefficients(n, n, degree, lg); // Coefficients for the Psi Polynomial System
	// privateKey.phi = phiCoefficients;
	// privateKey.psi = psiCoefficients;
	
	ll mPhi = MaxInCodomain(phiCoefficients, n, degree, l); // Largest Value in Codomain of Phi
	ll mPsi = MaxInCodomain(psiCoefficients, n, degree, l); // Largest Value in Codomain of Psi
	
	// TODO
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