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

int numHyper = 0;

//#define RANDOM_INITIAL
#define CHOLESKY
#define AC_EI
//#define FIX_VARIANCE

#define OPTIMIZE_EACH_SIZE
#define GAUSSIANKERNEL
//#define RATIONALQUADRATIC

const int Dimen = 2;
std::vector<double> sigmaF {0.250, 0.25}; // sample Gaussian function

double chi = 0.1; // used to weight acquistion function
double noise = 0.00001;



using namespace std;

int numP = 50;
int numP_max = 60;



double theta1sq_save = 0.02;
double deltasq_save = 0.1;
double delta2sq_save = 0.0;
double theta2sq_save = .0010;

double theta1sq = 0.35; // gaussian kernel
double deltasq = 0.10; // 

double delta2sq = delta2sq_save; // rational quadratic kernel parameters
double alpha = 12.0;
double theta2sq = theta2sq_save;

const double xmin = -1.0;
const double xmax = 1.0;
int seed = 1;
int plot_points = 500;

std::vector<std::vector<double> > trainingPoints(numP);
Eigen::VectorXd yp(numP); // y-values at training points



double funct(std::vector<double> x){
	//
	// multi-dimensional function to fit
	//
	int dimen = x.size();
	
	double exponent = 0.0;
	for (int i = 0;i< dimen;i++) exponent += x[i]*x[i]/sigmaF[i]/2.0;
	
	double f = exp(-exponent)*sin(3.0*M_PI*(x[0]-xmin)/(xmax-xmin) *cos(2.0*M_PI*(x[1]-xmin)/(xmax-xmin)));
	return f;
}

double gaussianKernel(double rsq){
	double rbf = deltasq*exp(-rsq/2.0/theta1sq);
	return rbf;
}

double rationalQuadraticKernel(double rsq){
	double x = 1.0 + rsq/alpha/theta2sq;
	double rqk = delta2sq*pow(x,-alpha);
	return rqk;
}


double covariance(std::vector<double> &x1, std::vector<double> &x2){
	int dimen = x1.size();
	if (x2.size() != dimen){
		std::cerr << "Error in size of training x set points." << std::endl;
		exit(1);
	}
	
	double dsq = 0.0;
	for (int i=0;i<dimen;i++){
		dsq += (x2[i]-x1[i])*(x2[i]-x1[i]);
	}
	double cov = 1.0;
	
#ifdef GAUSSIANKERNEL
	cov += gaussianKernel(dsq);
#endif

#ifdef RATIONALQUADRATIC
	cov += rationalQuadraticKernel(dsq);
#endif
	return cov;
}

void constructKmatrix(Eigen::MatrixXd & K){
	// kernel function
	for (int i=0;i<numP;i++){
		for (int j=i;j<numP;j++){
			K(i,j) = covariance(trainingPoints[i], trainingPoints[j]);
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


void drawMC(std::vector<double> &x, std::default_random_engine &localEngine){
	double f_old = fabs(funct(x));
	double dmax = 0.50;

	int numMC = 100;
	int numAccept = 0;
	for (int i=0;i<numMC;i++){
		double x0new = x[0] + pick_a_number(localEngine,-dmax, dmax);
		if ( (x0new > xmax) or (x0new < xmin) ) continue;
		
		double x1new = x[1] + pick_a_number(localEngine, -dmax, dmax);
		if ( (x1new > xmax) or (x1new < xmin) ) continue;
		
		std::vector<double> xnew = {x0new, x1new};
		double f_new = fabs(funct(xnew));
		double ratio = f_new/f_old;
		
		if (ratio > 1.0){
			f_old = f_new;
			x = xnew;
			numAccept++;
		} else if ( pick_a_number(localEngine, 0.0, 1.0) < ratio){
			f_old = f_new;
			x = xnew;
			numAccept++;
		}
		
	}
	double ratio = double(numAccept)/double(numMC);
	std::cerr << "  Drew point " << x[0] << " " << x[1] << "  from accept ratio = " << ratio << std::endl;
}

static double my_f(const gsl_vector *vec, void* tptr){
	deltasq = gsl_vector_get(vec,0);
	theta1sq = gsl_vector_get(vec,1);
#ifdef RATIONALQUADRATIC
	delta2sq = gsl_vector_get(vec,2);
	theta2sq = gsl_vector_get(vec,3);
#endif
	
	Eigen::MatrixXd Kxx(numP,numP);
	constructKmatrix(Kxx);

  	Eigen::LLT<Eigen::MatrixXd> lltOfK(Kxx); // compute the Cholesky decomposition of K
	Eigen::MatrixXd L = lltOfK.matrixL(); // retrieve factor L  in the decomposition
	Eigen::VectorXd b = lltOfK.solve(yp);
	double half_log_det = 0.0;
	for (int i=0;i<numP;i++) {
		if (L(i,i) > 0.0) half_log_det += log(L(i,i));
	}
	
	double log_likelihood = -0.5*yp.transpose()*b - half_log_det - numP/2*log(2.0*M_PI);

	//cout << " my_f:  deltasq = " << deltasq << "   theta1sq = " << theta1sq << "   -log_l = " << -log_likelihood << endl;
	return -log_likelihood;
}


double fn1 (double thetasq, void * params)
{
  (void)(params); /* avoid unused parameter warning */

	theta1sq = thetasq;
	Eigen::MatrixXd Kxx(numP,numP);
	constructKmatrix(Kxx);

#ifdef CHOLESKY
	Eigen::LLT<Eigen::MatrixXd> lltOfK(Kxx); // compute the Cholesky decomposition of K
	Eigen::MatrixXd L = lltOfK.matrixL(); // retrieve factor L  in the decomposition
	Eigen::VectorXd b = lltOfK.solve(yp);
	double two_log_det = 0.0;
	for (int i=0;i<numP;i++) {
		if (L(i,i) > 0.0) two_log_det += log(L(i,i));
		
	}
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

void adaptFit(std::vector<double>  &x_val, Eigen::MatrixXd & K, Eigen::MatrixXd &Kinv, Eigen::VectorXd & b){
	
	trainingPoints.push_back(x_val);

	numP++;
	
	yp.resize(numP);
	yp(numP-1) = funct(x_val);

	K.resize(numP,numP);
	Kinv.resize(numP,numP);
	b.resize(numP);
		
	cout << "  Inverse: New adapted x points are: "<< endl;
	for (int i=0;i<numP;i++) cout << trainingPoints[i][0] << " " << trainingPoints[i][1] << " " << yp(i) <<  endl;
	
	constructKmatrix(K);
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
	
	cout << "  Starting maximization with delta1sq = " << deltasq << "   theta1sq = " << theta1sq << "  num points = " << numP << endl;
	deltasq = delta2sq_save;
	//theta1sq = theta1sq_save;
	delta2sq = 0.0;
	//theta2sq = theta2sq_save; // start from initial values
	
	const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
	gsl_multimin_fminimizer *s = NULL;
	gsl_multimin_function minex_func;
   
	gsl_vector *ss;
	gsl_vector *x;

	/* Starting point */
#ifdef RATIONALQUADRATIC
	x = gsl_vector_alloc (4);
	gsl_vector_set (x, 0, deltasq);
	gsl_vector_set (x, 1, theta1sq);
	gsl_vector_set (x, 2, delta2sq);
	gsl_vector_set (x, 3, theta2sq);
	/* Set initial step sizes to 0.01 */
	ss = gsl_vector_alloc (4);
	gsl_vector_set_all (ss, 0.010);
	minex_func.f = my_f;	
	minex_func.n = 4;
	s = gsl_multimin_fminimizer_alloc (T, 4);
#else
	x = gsl_vector_alloc (2);
	gsl_vector_set (x, 0, deltasq);
	gsl_vector_set (x, 1, theta1sq);

	/* Set initial step sizes to 0.01 */
	ss = gsl_vector_alloc (2);
	gsl_vector_set_all (ss, 0.010);
	minex_func.f = my_f;
	minex_func.n = 2;
	s = gsl_multimin_fminimizer_alloc (T, 2);
#endif
  /* Initialize method and iterate */
	

	minex_func.params = NULL;

	
	gsl_multimin_fminimizer_set (s, &minex_func, x, ss);

	do
    {
		iter++;
		status = gsl_multimin_fminimizer_iterate(s);
		
		if (status) 
		  break;

		size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, 1e-3);

		if (status == GSL_SUCCESS)
        {
			printf ("converged to minimum at\n");
        }

    }
	while (status == GSL_CONTINUE && iter < 200);
  	deltasq = gsl_vector_get(s->x, 0);
	theta1sq = gsl_vector_get(s->x, 1);
#ifdef RATIONALQUADRATIC
	delta2sq = gsl_vector_get(s->x, 2);
	theta2sq = gsl_vector_get(s->x, 3);
#endif

#ifdef RATIONALQUADRATIC

	printf ("%5d %10.3e %10.3e %10.3e %10.3e f() = %7.3f size = %.3f\n", 
				iter,
				gsl_vector_get (s->x, 0), 
				gsl_vector_get (s->x, 1), 
				gsl_vector_get (s->x, 2),
				gsl_vector_get (s->x, 3),
				s->fval, size);

#else
		printf ("%5d %10.3e %10.3e f() = %7.3f size = %.3f\n", 
				iter,
				gsl_vector_get (s->x, 0), 
				gsl_vector_get (s->x, 1), 
				s->fval, size);

#endif



	gsl_vector_free(x);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free (s);

	//deltasq = 1.0;

	return status;
}

void adaptFitCholesky(std::vector<double>& x_val, Eigen::MatrixXd & K, Eigen::MatrixXd & L, Eigen::VectorXd & b){
	
	trainingPoints.push_back(x_val); // expand size of training set
	Eigen::VectorXd ypsave = yp;
	numP++;
	
	int trainingSize = trainingPoints.size();
	if (trainingSize != numP){
		std::cerr << "  Training size should be " << numP << " and is " << trainingSize << std::endl;
		exit(1);
	}
	
	yp.resize(numP);
	
	for (int i=0;i<numP;i++) yp(i) = funct(trainingPoints[i]);
	
	K.resize(numP,numP);
	L.resize(numP,numP);
	b.resize(numP);
		
	//cout << "  Cholesky:  New adapted " << numP << " x points are: "<< endl;
	//for (int i=0;i<numP;i++) cout << trainingPoints[i][0] << " " << trainingPoints[i][1] << " " << yp(i) <<  endl;

	
#ifdef OPTIMIZE_EACH_SIZE
#ifdef FIX_VARIANCE
	maximizeLength(); // 1D Optimization of length only
#else
	maximizeHyperParameters();
#endif
#endif
	

	constructKmatrix(K);
	Eigen::LLT<Eigen::MatrixXd> lltOfK(K); // compute the Cholesky decomposition of K
	L = lltOfK.matrixL(); // retrieve factor L  in the decomposition
	b = lltOfK.solve(yp);


}

bool pointInSet(std::vector<double> &xv){
	bool returnFlag = false;
	for (int i=0;i<trainingPoints.size();i++){
		double diff0 = xv[0] - trainingPoints[i][0];
		double diff1 = xv[1] - trainingPoints[i][1];
		double magsq = diff0*diff0 + diff1*diff1;
		if (magsq < 0.0000001) returnFlag = true;
	}
	return returnFlag;
	
}
int main(int argc,char** argv){
	std::default_random_engine localEngine;
	localEngine.seed( seed );

	std::ofstream pointsFile("points.dat");
	
	//trainingPoints[0].push_back(xmin);
	//trainingPoints[0].push_back(xmin);
	
	//trainingPoints[1].push_back(xmax);
	//trainingPoints[1].push_back(xmax);

	//yp(0) = funct(trainingPoints[0]);
	//yp(1) = funct(trainingPoints[1]);
	
	//  Get training points

#ifdef RANDOM_INITIAL
	for (int i=0;i<numP;i++){
		trainingPoints[i].push_back( pick_a_number(localEngine, xmin, xmax) );
		trainingPoints[i].push_back( pick_a_number(localEngine, xmin, xmax) );
		yp(i) = funct( trainingPoints[i] );
	
	}
#else
	for (int i=0;i<numP;i++){
	    double x0 = pick_a_number(localEngine, xmin, xmax);
	    double x1 = pick_a_number(localEngine, xmin, xmax);
	    std::vector<double> xtrial = {x0, x1};
	    
	    drawMC(xtrial, localEngine);
	    
		trainingPoints[i].push_back( xtrial[0] );
		trainingPoints[i].push_back( xtrial[1] );
		yp(i) = funct( trainingPoints[i] );
	
	}

#endif
	cout << " x and y points are:" << endl;
	for (int i=0;i<numP;i++) {
		cout << trainingPoints[i][0] << " " << trainingPoints[i][1] << " " << yp(i) << std::endl;
		pointsFile << trainingPoints[i][0] << " " << trainingPoints[i][1] << " " << yp(i) << std::endl;
	}
	//  Make kernel, covariance matrix
	Eigen::MatrixXd Kxx(numP,numP);
	constructKmatrix(Kxx);
	
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
		constructKmatrix(Kxx);
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
	theta1sq = theta1sq_save;
#ifdef FIX_VARIANCE
	maximizeLength(); // 1D Optimization only
#else
	maximizeHyperParameters();
#endif

	constructKmatrix(Kxx);

#ifdef CHOLESKY
	Eigen::LLT<Eigen::MatrixXd> lltOfK(Kxx); // compute the Cholesky decomposition of K
	Eigen::MatrixXd L = lltOfK.matrixL(); // retrieve factor L  in the decomposition
	b_coeff = lltOfK.solve(yp);
	double max_b = b_coeff.maxCoeff();
	cout << "   Parameters are  deltasq= " << deltasq << "   theta1sq = " << theta1sq;
#ifdef RATIONALQUADRATIC
	cout << "  delta1sq = " << delta2sq << "    theta2sq = " << theta2sq;
#endif
	cout << endl;

	
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
		
		ofstream dataFile("fit.dat");
		//
		double fmax = -100000000000;
		double fmin = 10000000000;
        std::vector<double> x_star = trainingPoints[0];
        for (int i=0;i<numP;i++){
			double fi = funct(trainingPoints[i]);
			if (fi > fmax){
				fmax = fi;
				x_star = trainingPoints[i];
			}
			if (fi < fmin){
				fmin = fi;
			}
		}
		//
		double ac_max = 0.0;
		std::vector<double> x_variance = trainingPoints[0];
		double av_sq_err = 0.0;
		
		
		
		for (int ix=0;ix<=plot_points;ix++){
			for (int iy=0;iy <= plot_points;iy++){
				std::vector<double> xv  = {xmin + ix*dx, xmin + iy*dx};
				
				double yv = funct(xv);
			
				Eigen::VectorXd k_vec(numP);
				for (int j=0;j<numP;j++) k_vec(j) = covariance(xv, trainingPoints[j]);

			
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
				
				plotFile << xv[0] << " " << xv[1] << " " << yv << " " << y_predicted << " " << variance << " " << acquisition
					 << " " << fabs(yv - y_predicted) << " " << k_vec.maxCoeff() << std::endl;
			    dataFile << xv[0] << " " << xv[1] << " " << yv << " " << y_predicted << std::endl;
				av_sq_err += (yv-y_predicted)*(yv-y_predicted);
			}
		}
		av_sq_err /= double(plot_points*plot_points);
		
		cout << "  Maximum of acquisition occurs at " << x_variance[0] << " " << x_variance[1] << "  and is " << ac_max << "  err = " << av_sq_err << endl;
		if (av_sq_err < (0.00001+sqrt(noise))) {
			cout << "  Converged with num points = " << numP << endl;
			break;
		}
		plotFile.close();
		
		if ( pointInSet(x_variance) ){
			std::cerr << "  Point of max. variance already in set.  Using random point instead." << endl;		
			drawMC(x_variance, localEngine);		
			//x_variance[0] = pick_a_number(localEngine, xmin, xmax);
			//x_variance[1] = pick_a_number(localEngine, xmin, xmax);
		} 
		
		if (ac_max < 0.003) drawMC(x_variance, localEngine); // sample another point
		
		
#ifdef CHOLESKY
		adaptFitCholesky(x_variance, Kxx, L, b_coeff);
#else
		adaptFit(x_variance, Kxx, Kinv, b_coeff);
#endif
	}
	std::cout << "  Final fit with " << numP << " points:" << std::endl;
	for (int i=0;i<numP;i++) cout << trainingPoints[i][0] << " " << trainingPoints[i][1] << " " << yp(i) <<  endl;

	//
	

	
	return 1;
}

