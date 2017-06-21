#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define EPS 0.001

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	is_initialized_ = false;
	
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.49;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.52;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  n_x_ = x_.size();

	n_aug_ = n_x_ + 2; // We will create 2 * n_aug_ + 1 sigma points 

	// Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;
	
  // Set the predicted sigma points matrix dimentions
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
	
  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;
	
  // Weights of sigma points
  weights_ = VectorXd(n_sig_);
	
  // Measurement noise covariance matrices initialization
  R_radar_ = MatrixXd(3, 3);
	
  R_radar_ << std_radr_*std_radr_, 0, 0,
		0, std_radphi_*std_radphi_, 0,
		0, 0,std_radrd_*std_radrd_;
	
  R_lidar_ = MatrixXd(2, 2);
	
  R_lidar_ << std_laspx_*std_laspx_,0,
		0,std_laspy_*std_laspy_;
	
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage &meas_package) {
	  if (!is_initialized_) { 
			/**
				 TODO:
				 * Initialize the state x_ with the first measurement.
				 * x_ is [px, py, vel, ang, ang_rate]
				 */
			// first measurement
			cout << "UKF: " << endl;
			x_ = VectorXd(5);
			
			P_ << 1, 0, 0, 0, 0,
				0, 1, 0, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 0, 1;
			
			if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
				/**
					 Convert radar from polar to cartesian coordinates and initialize state.
				*/
				float rho = meas_package.raw_measurements_[0];
				// range
				float phi = meas_package.raw_measurements_[1];
				// bearing
				float rho_dot = meas_package.raw_measurements_[2];
				// velocity of rho
				// Coordinates convertion from polar to cartesian
				float px = rho * cos(phi); 
				float py = rho * sin(phi);
				float vx = rho_dot * cos(phi);
				float vy = rho_dot * sin(phi);
				float v = sqrt(vx * vx + vy * vy);
				cout << "Use initialization as radar" << endl;
				x_ << px, py, v , 0, 0;
			}
			else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
				
				/**
					 Initialize state.
				*/
				x_ << meas_package.raw_measurements_[0], 
					meas_package.raw_measurements_[1], 
					0,
					0, 
					0; 

				cout << "Use initialization as lidar" << endl;
			}
			time_us_ = meas_package.timestamp_;
		
			// Initialize weights
			weights_(0) = lambda_ / (lambda_ + n_aug_);
			for (int i = 1; i < weights_.size(); i++) {
				weights_(i) = 0.5 / (n_aug_ + lambda_);
			}
			// done initializing, no need to predict or update
			is_initialized_ = true;
			return;
		}

		//compute the time elapsed between the current and previous
		//measurements
		float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
		//dt - expressed in seconds
		time_us_ = meas_package.timestamp_;

		Prediction(dt);

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
			UpdateRadar(meas_package);
		}

		if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
			UpdateLidar(meas_package);
		}
}

void NormAng(double *ang) 
{
	while (*ang > M_PI) *ang -= 2. * M_PI;	
	while (*ang < -M_PI) *ang += 2. * M_PI;
}

void UKF::GenerateSigmaPoints(MatrixXd& Xsig)
{
	// Mean vector
  VectorXd x = VectorXd(n_aug_);
  // State covarience matrix
  MatrixXd P = MatrixXd(n_aug_, n_aug_);

	x.fill(0.0);
  x.head(n_x_) = x_;

  P.fill(0);
  P.topLeftCorner(n_x_,n_x_) = P_;
  P(5,5) = std_a_*std_a_;
  P(6,6) = std_yawdd_*std_yawdd_;
	
  // Square root of P matrix
  MatrixXd A = P.llt().matrixL();

  // First sigma point
  Xsig.col(0) = x;
	
  double sqrt_lambda_n_aug = sqrt(lambda_+n_aug_);
  VectorXd sqrt_lambda_n_aug_A;
	// The rest sigma points
  for(int i = 0; i < n_aug_; i++) {
		sqrt_lambda_n_aug_A = sqrt_lambda_n_aug * A.col(i);
    Xsig.col(i+1)        = x + sqrt_lambda_n_aug_A;
    Xsig.col(i+1+n_aug_) = x - sqrt_lambda_n_aug_A;
  }
}

void UKF::PredictSigmaPoints(double delta_t, const MatrixXd& Xsig) 
{
	double delta_t2 = delta_t*delta_t;

	for (int i = 0; i< n_sig_; i++)
  {
    double p_x = Xsig(0,i);
    double p_y = Xsig(1,i);
    double v = Xsig(2,i);
    double yaw = Xsig(3,i);
    double yawd = Xsig(4,i);
    double nu_a = Xsig(5,i);
    double nu_yawdd = Xsig(6,i);

    double sin_yaw = sin(yaw);
    double cos_yaw = cos(yaw);
    double arg = yaw + yawd*delta_t;
		double px_p, py_p;

    if (fabs(yawd) > EPS) {
			double v_yawd = v/yawd;
			px_p = p_x + v_yawd * (sin(arg) - sin_yaw);
			py_p = p_y + v_yawd * (cos_yaw - cos(arg) );
    }
    else {
			double v_delta_t = v*delta_t;
			px_p = p_x + v_delta_t*cos_yaw;
			py_p = p_y + v_delta_t*sin_yaw;
    }
		
    double v_p = v;
    double yaw_p = arg;
    double yawd_p = yawd;

    px_p += 0.5*nu_a*delta_t2*cos_yaw;
    py_p += 0.5*nu_a*delta_t2*sin_yaw;
    v_p += nu_a*delta_t;
    yaw_p += 0.5*nu_yawdd*delta_t2;
    yawd_p += nu_yawdd*delta_t;

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() 
{
	x_ = Xsig_pred_ * weights_;
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormAng(&(x_diff(3)));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
	}
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
    Estimates the object's location. Modify the state
		vector, x_. Predicts sigma points, the state, and the state covariance matrix.
  */
	MatrixXd Xsig = MatrixXd(n_aug_, n_sig_);
  
	GenerateSigmaPoints(Xsig);
	PredictSigmaPoints(delta_t, Xsig);
	PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage &meas_package) {
	// Set measurement dimension
  int n_z = 2;
	
  // Create matrix for sigma points in measurement space
  // Transform sigma points into measurement space
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);
	
	UpdateUKF(meas_package, Zsig, n_z);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage &meas_package) {

	// Set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  // Sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
	
  // Transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {
    
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
		
    //r
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);

		//phi
    Zsig(1,i) = atan2(p_y,p_x);

		//r_dot
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / Zsig(0,i);
  }
	
	UpdateUKF(meas_package, Zsig, n_z);
}


// Universal update function
void UKF::UpdateUKF(const MeasurementPackage &meas_package, const MatrixXd &Zsig, int n_z)
{
	
  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred  = Zsig * weights_;
	
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
	
  for (int i = 0; i < n_sig_; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Angle normalization
    NormAng(&(z_diff(1)));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
	
  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
	
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
		// Radar
    R = R_radar_;
  }
	
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
		// Lidar
    R = R_lidar_;
  }
	
  S = S + R;
  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  // Calculate cross correlation matrix
  Tc.fill(0.0);
	
  for (int i = 0; i < n_sig_; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
			// Radar
      // Angle normalization
      NormAng(&(z_diff(1)));
    }
		
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    NormAng(&(x_diff(3)));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
	
  // Measurements
  VectorXd z = meas_package.raw_measurements_;
  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  // Residual
  VectorXd z_diff = z - z_pred;
	
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
		// Radar
    // Angle normalization
    NormAng(&(z_diff(1)));
  }
	
  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
	
  // Calculate NIS
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
		// Radar
		NIS_radar_ = z.transpose() * S.inverse() * z;
  }
	
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
		// Lidar
		NIS_laser_ = z.transpose() * S.inverse() * z;
  }	
}
