#include <bits/stdc++.h>

#define ll long long int

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
	ll num = 1;
	for(ll i = 1; i <= d; ++i) {
		ll c = Combinations(n + i - 1, i);
		num += c;
	}
	return num;
}

typedef struct PublicKey {
	ll dummy;
	// TODO
} PublicKey;

typedef struct PrivateKey {
	ll dummy;
	// TODO
} PrivateKey;

PublicKey publicKey; // Public Key
PrivateKey privateKey; // Private Key

// Generate Public Private Key Pair
void GenerateKeyPair(ll n, ll l, ll lg, ll degree) {
	// TODO
	ll numMonomials = NumberOfMonomials(n, degree);
	std::vector<std::vector<ll>> coefficients; // List of Coefficients for the Polynomial
}

int main() {
	
	std::cout << "Enter Security Parameters (n L Lg degree):\n";
	ll n; // Number of Variables in System of Polynomial Equations
	ll l; // Odd Positive Integer
	ll lg; // Odd Positive Integer
	ll degree; // Maximum Degree of the Monomials in the Polynomial System
	std::cin >> n >> l >> lg >> degree;

	GenerateKeyPair(n, l, lg, degree);

	std::cout << "Enter Input to Encrypt (Enter " << n << " numbers in the range [" << std::floor(-l / 2.0) + 1 << ", " << std::floor(l / 2.0) << "]):\n";
	ll msg[n]; // Input to Encrypt
	for(ll i = 0; i < n; ++i) {
		std::cin >> msg[i];
	}

	// std::cout << NumberOfMonomials(n, degree) << "\n";

	return 0;
}