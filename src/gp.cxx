#include <iostream>
#include <math.h>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_vector.h>

//
//    Based on book found in reprints/GaussianProcess/RW2.pdf and RW5.pdf for maximization
//

#define CHOLESKY
//#define AC_EI
//#define FIX_VARIANCE
#define OPTIMIZE_EACH_SIZE

double chi = 0.1; // used to weight acquistion function
double noise = 0.0000001;

double v0 = 1.0;
double barrier = 0.750;
const double a0 = 10000.0;
const double b0 = 2.0;
const double x0 = -2.0;
const double a1 = 2.50;
//const double b1 = 6.0;
const double b1 = 16.0;
const double x1 = 2.0;




using namespace std;

int numP = 8;
int numP_max = 50;

double theta1sq = 0.35;
double deltasq = 0.10;

const double xmin = x0;
const double xmax = x1;
int seed = 1;
int plot_points = 500;

Eigen::VectorXd xp(numP);
Eigen::VectorXd yp(numP);


double funct(double x){
	double prefactor = b0*b1/sqrt(M_PI)/(a0*b1+a1*b0);
	double anal = prefactor*(a0/sqrt(b0)*exp(-b0*(x-x0)*(x-x0)) + a1/sqrt(b1)*exp(-b1*(x-x1)*(x-x1)));

	double f = - log(anal);
	return f;
}




double covariance(double x1, double x2){
	double dsq = (x2-x1)*(x2-x1)/2.0/theta1sq;
	double cov = deltasq*exp(-dsq);
	return cov;
}

void constructKmatrix(Eigen::MatrixXd & K, Eigen::VectorXd &xp){
	for (int i=0;i<numP;i++){
		for (int j=i;j<numP;j++){
			K(i,j) = covariance(xp(i), xp(j));
			K(j,i) = K(i,j);
		}
		K(i,i) += noise;
	}

}

double pick_a_number(std::default_random_engine &localEngine, double from, double upto )
{
    std::uniform_real_distribution<double> d(from, upto);
	return d(localEngine);
}

static double my_f(const gsl_vector *vec, void* tptr){
	deltasq = gsl_vector_get(vec,0);
	//deltasq = 1.0;
	theta1sq = gsl_vector_get(vec,1);
	
	Eigen::MatrixXd Kxx(numP,numP);
	constructKmatrix(Kxx, xp);

  	Eigen::LLT<Eigen::MatrixXd> lltOfK(Kxx); // compute the Cholesky decomposition of K
	Eigen::MatrixXd L = lltOfK.matrixL(); // retrieve factor L  in the decomposition
	Eigen::VectorXd b = lltOfK.solve(yp);
	double half_log_det = 0.0;
	for (int i=0;i<numP;i++) half_log_det += log(L(i,i));
	double log_likelihood = -0.5*yp.transpose()*b - half_log_det - numP/2*log(2.0*M_PI);

	cout << " my_f:  deltasq = " << deltasq << "   theta1sq = " << theta1sq << "   -log_l = " << -log_likelihood << endl;
	return -log_likelihood;
}


double fn1 (double thetasq, void * params)
{
  (void)(params); /* avoid unused parameter warning */

  theta1sq = thetasq;
  Eigen::MatrixXd Kxx(numP,numP);
  constructKmatrix(Kxx, xp);

#ifdef CHOLESKY
  	Eigen::LLT<Eigen::MatrixXd> lltOfK(Kxx); // compute the Cholesky decomposition of K
	Eigen::MatrixXd L = lltOfK.matrixL(); // retrieve factor L  in the decomposition
	Eigen::VectorXd b = lltOfK.solve(yp);
	double two_log_det = 0.0;
	for (int i=0;i<numP;i++) two_log_det += log(L(i,i));
	double log_likelihood = -0.5*yp.transpose()*b - two_log_det - numP/2*log(2.0*M_PI);

	// Eigen::MatrixXd Kinv = Kxx.inverse();
	// double determinant = Kxx.determinant();
	// double log_likelihood_test = -0.5*yp.transpose()*Kinv*yp - 0.5*log(determinant) - numP/2*log(2.0*M_PI);
	// cout << "  log_likelihood = " << log_likelihood << " = " << log_likelihood_test << endl;
#else
  Eigen::MatrixXd Kinv = Kxx.inverse();
  double determinant = Kxx.determinant();
  double log_likelihood = -0.5*yp.transpose()*Kinv*yp - 0.5*log(determinant) - numP/2*log(2.0*M_PI);
#endif
  //cout << "  thetasq = " << thetasq << "   -log_l = " << -log_likelihood << endl;
  return -log_likelihood;
}

void adaptFit(double x_val, Eigen::MatrixXd & K, Eigen::MatrixXd &Kinv, Eigen::VectorXd & b){
	Eigen::VectorXd xpsave = xp;
	Eigen::VectorXd ypsave = yp;
	numP++;
	xp.resize(numP);
	xp(numP-1) = x_val;
	
	yp.resize(numP);
	yp(numP-1) = funct(x_val);

	for (int i=0;i<numP-1;i++){
		xp(i) = xpsave(i);
		yp(i) = ypsave(i);
	}
	
	K.resize(numP,numP);
	Kinv.resize(numP,numP);
	b.resize(numP);
		
	cout << "  Inverse: New adapted x points are: "<< endl;
	for (int i=0;i<numP;i++) cout << xp(i) << " " << yp(i) <<  endl;
	
	constructKmatrix(K, xp);
	Kinv = K.inverse();
	b = Kinv*yp;

}



int maximizeLength(){
	//
	//  Minimize log-likelihood with fixed variance: 1D optimization
	//
	//theta1sq = 0.3;
	int status;
	int iter = 0, max_iter = 100;
	const gsl_min_fminimizer_type *T;
	gsl_min_fminimizer *s;
	
	double a = 0.01, b = 2.50; // range of variance to find max
	double epsilon = 0.0001;
	gsl_function F;

	F.function = &fn1;
	F.params = 0;

	double theta1sq_save = theta1sq; // save original value if bracketing fails

	double f_a = fn1(a,F.params);
	double f_b = fn1(b,F.params);
	
	double m = a + epsilon; // initial guess of variance below value of endpoints
	if ( f_b < f_a ) m = b - epsilon;
	double f_m = fn1(m,F.params);
	if ((f_m > f_a) or (f_m > f_b) ){
		cerr << "  Bracketing failed since f_m  not below end points "
			 << f_m << " at x = " << m << " not less than both initial values " << f_a << " " << f_b << endl;
		theta1sq = theta1sq_save;
	} else {
		cout << "  Minimizer bracketing: " << f_a << " > " << fn1(m,F.params) << " < " <<  f_b << endl;
	
		T = gsl_min_fminimizer_brent;
		//T = gsl_min_fminimizer_goldensection;
		s = gsl_min_fminimizer_alloc (T);
		gsl_min_fminimizer_set (s, &F, m, a, b);

		cout << "Entering minimization with method " << gsl_min_fminimizer_name(s) << endl;
		do
		{
			iter++;
			status = gsl_min_fminimizer_iterate (s);

			m = gsl_min_fminimizer_x_minimum (s);
			a = gsl_min_fminimizer_x_lower (s);
			b = gsl_min_fminimizer_x_upper (s);
		
			status = gsl_min_test_interval (a, b, 0.0001, 0.0);
			
			if (status == GSL_SUCCESS)  cout << " Converged:" << endl;
			
			cout << iter << " " << a << " " << b
				 << " " << m << endl;
		}
		while (status == GSL_CONTINUE && iter < max_iter);
		theta1sq = m;
		gsl_min_fminimizer_free (s);
	}
	//
	// Now check fit
	double fval = fn1 (theta1sq,F.params);
	cout << "  Fit for optimal variance = " << theta1sq << "   -log_l = " << fval << endl;
	return status;
}
int maximizeHyperParameters(){
	//
	//  Minimize log-likelihood
	//
	int status;
	int iter = 0, max_iter = 100;
	double size;
	
	const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
	gsl_multimin_fminimizer *s = NULL;
	gsl_multimin_function minex_func;
   
	gsl_vector *ss;
	gsl_vector *x;

	/* Starting point */
	x = gsl_vector_alloc (2);
	gsl_vector_set (x, 0, deltasq);
	gsl_vector_set (x, 1, theta1sq);

	/* Set initial step sizes to 0.01 */
	ss = gsl_vector_alloc (2);
	gsl_vector_set_all (ss, 0.010);

  /* Initialize method and iterate */
	minex_func.n = 2;
	minex_func.f = my_f;
	minex_func.params = NULL;

	s = gsl_multimin_fminimizer_alloc (T, 2);
	gsl_multimin_fminimizer_set (s, &minex_func, x, ss);

	do
    {
		iter++;
		status = gsl_multimin_fminimizer_iterate(s);
		
		if (status) 
		  break;

		size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, 1e-2);

		if (status == GSL_SUCCESS)
        {
			printf ("converged to minimum at\n");
        }

		printf ("%5d %10.3e %10.3e f() = %7.3f size = %.3f\n", 
				iter,
				gsl_vector_get (s->x, 0), 
				gsl_vector_get (s->x, 1), 
				s->fval, size);
    }
	while (status == GSL_CONTINUE && iter < 100);
  	deltasq = gsl_vector_get(s->x, 0);
	theta1sq = gsl_vector_get(s->x, 1);
	
	gsl_vector_free(x);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free (s);

	//deltasq = 1.0;

	return status;
}

void adaptFitCholesky(double x_val, Eigen::MatrixXd & K, Eigen::MatrixXd & L, Eigen::VectorXd & b){
	Eigen::VectorXd xpsave = xp;
	Eigen::VectorXd ypsave = yp;
	numP++;
	xp.resize(numP);
	xp(numP-1) = x_val;
	
	yp.resize(numP);
	yp(numP-1) = funct(x_val);

	for (int i=0;i<numP-1;i++){
		xp(i) = xpsave(i);
		yp(i) = ypsave(i);
	}
	
	K.resize(numP,numP);
	L.resize(numP,numP);
	b.resize(numP);
		
	cout << "  Cholesky:  New adapted " << numP << " x points are: "<< endl;
	for (int i=0;i<numP;i++) cout << xp(i) << " " << yp(i) <<  endl;

	
#ifdef OPTIMIZE_EACH_SIZE
#ifdef FIX_VARIANCE
	maximizeLength(); // 1D Optimization only
#else
	maximizeHyperParameters();
#endif
#endif
	

	constructKmatrix(K, xp);
	Eigen::LLT<Eigen::MatrixXd> lltOfK(K); // compute the Cholesky decomposition of K
	L = lltOfK.matrixL(); // retrieve factor L  in the decomposition
	b = lltOfK.solve(yp);


}

int main(int argc,char** argv){
	std::default_random_engine localEngine;
	localEngine.seed( seed );

	std::ofstream pointsFile("points.dat");
	xp(0) = xmin;
	xp(1) = xmax;
	yp(0) = funct(xmin);
	yp(1) = funct(xmax);
	
	//  Get training points

	for (int i=2;i<numP;i++){
		xp(i) = pick_a_number(localEngine, xmin, xmax);
		yp(i) = funct( xp(i) );
	
	}
	cout << " x and y points are:" << endl;
	for (int i=0;i<numP;i++) {
		cout << xp(i) << " " << yp(i) << std::endl;
		pointsFile << xp(i) << " " << yp(i) << std::endl;
	}
	//  Make kernel, covariance matrix
	Eigen::MatrixXd Kxx(numP,numP);
	constructKmatrix(Kxx, xp);
	
	Eigen::MatrixXd Kinv = Kxx.inverse();
	Eigen::VectorXd b_coeff = Kinv*yp;

	//

	// Now plot log-likelihood as a function of theta1sq for fixed number of points
        // log-likelihood for Gaussian process, eq 5.8 (or 2.30) of text
	//
	ofstream logL("logl.dat");
	double theta1sq_min = 0.001;
	double theta1sq_max = 2.0;
	double dtheta = (theta1sq_max - theta1sq_min)/double(plot_points);
	for (int i=0;i<=plot_points;i++){
		theta1sq = theta1sq_min + i*dtheta;
		constructKmatrix(Kxx, xp);
		Eigen::LLT<Eigen::MatrixXd> lltOfK(Kxx); // compute the Cholesky decomposition of K
		Eigen::MatrixXd L = lltOfK.matrixL(); // retrieve factor L  in the decomposition
		Eigen::VectorXd b = lltOfK.solve(yp);
		double two_log_det = 0.0;
		double det = 1.0;
		for (int i=0;i<numP;i++) {
			two_log_det += log(L(i,i));
			det *= L(i,i)*L(i,i);
		}
		double log_likelihood = -0.5*yp.transpose()*b - two_log_det - numP/2*log(2.0*M_PI);
		
		logL << theta1sq << " " << -log_likelihood  << " " << two_log_det << "  " << 0.5*log(det) << endl;

	}
	//

#ifdef FIX_VARIANCE
	maximizeLength(); // 1D Optimization only
#else
	maximizeHyperParameters();
#endif

	constructKmatrix(Kxx, xp);

#ifdef CHOLESKY
	Eigen::LLT<Eigen::MatrixXd> lltOfK(Kxx); // compute the Cholesky decomposition of K
	Eigen::MatrixXd L = lltOfK.matrixL(); // retrieve factor L  in the decomposition
	b_coeff = lltOfK.solve(yp);
	double max_b = b_coeff.maxCoeff();
	cout << "   Parameters are  deltasq= " << deltasq << "   theta1sq = " << theta1sq << endl;

	
#else		
	Kinv = Kxx.inverse();
	b_coeff = Kinv*yp;
#endif
	
	//
	double dx = (xmax-xmin)/double(plot_points);
	while (numP < numP_max){

		char* fileName = new char[20];
		sprintf(fileName,"fit%dp.dat",numP);
		ofstream plotFile(fileName);
		delete[] fileName;
		//
		double fmax = -100000000000;
		double fmin = 10000000000;
        double x_star = xmin;
        for (int i=0;i<numP;i++){
			double fi = funct(xp(i));
			if (fi > fmax){
				fmax = fi;
				x_star = xp(i);
			}
			if (fi < fmin){
				fmin = fi;
			}
		}
		//
		double ac_max = 0.0;
		double x_variance = xmin;
		double av_sq_err = 0.0;
		for (int i=0;i<=plot_points;i++){
			double xv = xmin + i*dx;
			double yv = funct(xv);
			
			Eigen::VectorXd k_vec(numP);
			for (int j=0;j<numP;j++) k_vec(j) = covariance(xv, xp(j));

			
			double y_predicted = k_vec.transpose() * b_coeff;
#ifdef CHOLESKY	   
			Eigen::VectorXd v = L.triangularView<Eigen::Lower>().solve(k_vec);
			double variance = covariance(xv,xv) - v.squaredNorm();
#else
			double variance = covariance(xv,xv) - k_vec.transpose() * Kinv * k_vec;
			
#endif

#ifdef AC_EI
			double z = 0.0;
			double std_dev = sqrt(variance);
			if (variance > 0.0) z = (y_predicted - fmax - chi)/std_dev;
			double acquisition = variance*(z*gsl_cdf_gaussian_P(z,1.0) + gsl_ran_gaussian_pdf(z,1.0) );
#else
			double acquisition = variance;
#endif
			if (acquisition > ac_max){
				x_variance = xv;
				ac_max = acquisition;
			}
			plotFile << xv << " " << yv << " " << y_predicted << " " << variance << " " << acquisition
					 << " " << fabs(yv - y_predicted) << " " << k_vec.maxCoeff() << std::endl;
			av_sq_err += (yv-y_predicted)*(yv-y_predicted);
		}
		av_sq_err /= double(plot_points);
		cout << "  Maximum of acquisition occurs at " << x_variance << "  and is " << ac_max << "  err = " << av_sq_err << endl;
		if (av_sq_err < (0.0001+sqrt(noise))) {
			cout << "  Converged with num points = " << numP << endl;
			break;
		}
		plotFile.close();
#ifdef CHOLESKY
		adaptFitCholesky(x_variance, Kxx, L, b_coeff);
#else
		adaptFit(x_variance, Kxx, Kinv, b_coeff);
#endif
	}


	//
	

	
	return 1;
}

