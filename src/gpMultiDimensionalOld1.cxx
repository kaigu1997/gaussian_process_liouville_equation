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
int numKernel = 0;

const int Dimen = 2;
enum kernelType { gaussian, rational, exponential };

std::vector<double> hyperMag;
std::vector<double> hyperMagStart;
std::vector<double> hyperLengthSq[Dimen];
std::vector<double> hyperLengthSqStart[Dimen];
std::vector<kernelType> hyperType;

//#define RANDOM_INITIAL
#define CHOLESKY
#define AC_EI


#define OPTIMIZE_EACH_SIZE

//   Flag the kernel types
#define GAUSSIANKERNEL
//#define RATIONALQUADRATIC
#define EXPONENTIAL

//#define RESTARTOPTIMIZATION

std::vector<double> sigmaF {0.25, 0.25}; // sample Gaussian function

double chi = 0.1; // used to weight acquistion function
double noise = 0.00;

double alpha = 4.0; // fixed power for rational quadratic kernel
using namespace std;

int numP = 20;
int numP_max = 30;


const double xmin = -2.0;
const double xmax = 2.0;
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

double gaussianKernel(double rsq,double mag, double Lsq){
	double rbf = mag*exp(-rsq/2.0/Lsq);
	return rbf;
}

double rationalQuadraticKernel(double rsq,double mag, double Lsq){
	double x = 1.0 + rsq/alpha/Lsq;
	double rqk = mag*pow(x,-alpha);
	return rqk;
}

double exponentialKernel(double rsq, double mag, double Lsq){
	double r = sqrt(rsq);
	double ek = mag*exp(-r/Lsq);
	return ek;
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
	double cov = 0.0;
	
	for (int i=0;i<numHyper;i++){
		
		switch( hyperType[i] ) {
			case gaussian:
				cov += gaussianKernel(dsq, hyperMag[i], hyperLengthSq[i]);
				break;
			case rational:
				cov += rationalQuadraticKernel(dsq, hyperMag[i], hyperLengthSq[i]);
				break;
			case exponential:
				cov += exponentialKernel(dsq, hyperMag[i], hyperLengthSq[i]);
				break;
			default:
				std::cerr << "Unknown kernel type specified." << std::endl;
				exit(1);
		}	
	}
	
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
	
	
	for (int i=0;i<numHyper; i++){
		hyperMag[i] = gsl_vector_get(vec,i*2);
		hyperLengthSq[i] = gsl_vector_get(vec,i*2 + 1);
	}
	
	
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
	
	x = gsl_vector_alloc(2*numHyper);
	ss = gsl_vector_alloc(2*numHyper);
	
	gsl_vector_set_all (ss, 0.0050);
	minex_func.f = my_f;	
	minex_func.n = 2*numHyper;
	s = gsl_multimin_fminimizer_alloc (T, 2*numHyper);

#ifdef RESTARTOPTIMIZATION
	for (int i=0;i<numHyper;i++){
		hyperMag[i] = hyperMagStart[i];
		hyperLengthSq[i] = hyperLengthSqStart[i];
	}
#endif
	
	cout << " Starting minimization of log-likelihood for " << numHyper << " kernel types.  Initial parameters are:" << endl;
	for (int i=0;i<numHyper;i++){
		cout << " mag = " << hyperMag[i] << "    Lsq = " << hyperLengthSq[i] << " for type: ";
		switch( hyperType[i] ) {
			case gaussian:
				cout << "gaussian" << endl;
				gsl_vector_set(x,2*i,hyperMag[i]);
				gsl_vector_set(x,2*i+1,hyperLengthSq[i]);
				break;
			case rational:
				cout << "rational" << endl;
				gsl_vector_set(x,2*i,hyperMag[i]);
				gsl_vector_set(x,2*i+1,hyperLengthSq[i]);
				break;
			case exponential:
				cout << "exponential" << endl;
				gsl_vector_set(x,2*i,hyperMag[i]);
				gsl_vector_set(x,2*i+1,hyperLengthSq[i]);
				break;
			default:
				cout << "Not implemented...." << endl;
		}
			
	}

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
	while (status == GSL_CONTINUE && iter < 600);
	cout << "  Final hyper-parameters after " << iter << " steps are: " << endl;
	
	for (int i=0;i<numHyper;i++){
		hyperMag[i] = gsl_vector_get(s->x,2*i);
		hyperLengthSq[i] = gsl_vector_get(s->x,2*i+1);	
		cout << hyperMag[i] << " " << hyperLengthSq[i] << "    type: " << hyperType[i] << endl;
	}


	gsl_vector_free(x);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free (s);

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
	maximizeHyperParameters();
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

#ifdef GAUSSIANKERNEL
	numHyper++;
	hyperMagStart.push_back(0.14);
	hyperMag.push_back(0.14);
	
	hyperLengthSq.push_back(0.002);
	hyperLengthSqStart.push_back(0.002);
	hyperType.push_back(gaussian);
#endif

#ifdef RATIONALQUADRATIC
	numHyper++;
	hyperMag.push_back(0.2);
	hyperLengthSq.push_back(0.02);
	
	hyperMagStart.push_back(0.2);
	hyperLengthSqStart.push_back(0.02);
	hyperType.push_back(rational);
#endif

#ifdef EXPONENTIAL
	numHyper++;
	hyperMag.push_back(0.1);
	hyperLengthSq.push_back(0.10);
	
	hyperMagStart.push_back(0.1);
	hyperLengthSqStart.push_back(0.10);
	hyperType.push_back(exponential);
#endif

	std::ofstream pointsFile("points.dat");
	
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
	//
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
#else		
	Kinv = Kxx.inverse();
	b_coeff = Kinv*yp;
#endif
	
	//
	double dx = (xmax-xmin)/double(plot_points);
	double prevError = 100000000.0;
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

