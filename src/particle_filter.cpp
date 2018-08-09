/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	std::random_device rd{};
  std::mt19937 gen{rd()};

  std::normal_distribution<> dist_x{x, std[0]};
  std::normal_distribution<> dist_y{y, std[1]};
  std::normal_distribution<> dist_theta{theta, std[2]};

	num_particles = 30;
	for (int i = 0; i < num_particles; i++) {
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);
		Particle p;
		p.id = i;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.weight = 1;
		particles.push_back(p);
		weights.push_back(1);
		cout << "Sample " << i + 1 << " " << sample_x << " " << sample_y << " " << sample_theta << endl;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);

	// avoid divide by zero
	if (fabs(yaw_rate) < 0.00001) {
		yaw_rate = 0.00001;
	}

	for (int i = 0; i < num_particles; i++) {
		double x_f, y_f, theta_f;
		x_f = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
		y_f = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
		theta_f = particles[i].theta + yaw_rate * delta_t;
		particles[i].x = x_f + dist_x(gen);
		particles[i].y = y_f + dist_y(gen);
		particles[i].theta = theta_f + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {
		// find nearest associations
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;
		for (int j = 0; j < observations.size(); j++) {
			double x_m, y_m;
			x_m = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
			y_m = particles[i].y + sin(particles[i].theta) * observations[j].x - cos(particles[i].theta) * observations[j].y;
			double min_dist = 1e+90;
			int min_id;
			for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				double dist = pow(map_landmarks.landmark_list[k].x_f - x_m, 2) + pow(map_landmarks.landmark_list[k].y_f - y_m, 2);
				if (dist < min_dist) {
					min_dist = dist;
					min_id = k+1;
				}
			}

			associations.push_back(min_id);
			sense_x.push_back(x_m);
			sense_y.push_back(y_m);
		}

		SetAssociations(particles[i], associations, sense_x, sense_y);

		// calculate weight
		double weight = 1.0;
		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		for (int j = 0; j < observations.size(); j++) {
			int id = associations[j];
			double x = sense_x[i];
			double x_mu = map_landmarks.landmark_list[id-1].x_f;
			double y = sense_y[i];
			double y_mu = map_landmarks.landmark_list[id-1].y_f;
			double norm = 1.0 / (2 * M_PI * std_x * std_y);
			double exponent = pow(x - x_mu, 2.0) / 2.0 / pow(std_x, 2.0) + pow(y - y_mu, 2.0) / 2.0 / pow(std_y, 2.0);
			weight *=  norm * exp(-1.0 * exponent);
		}
		particles[i].weight = weight;
		weights[i] = weight;
		cout << std::scientific << "particles " << i << " weight: " << weight << endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<double> weights;
	cout << "before resample weights: ";
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
		cout << std::scientific << particles[i].weight << ", ";
	}
	cout << endl;

	std::random_device rd;
  std::mt19937 gen(rd());
	std::discrete_distribution<> d(weights.begin(), weights.end());
	std::vector<Particle> newParticles;
	cout << "resample index: ";
	for (int i = 0; i < num_particles; i++) {
		int index = d(gen);
		newParticles.push_back(particles[index]);
		cout << index << ", ";
	}
	cout << endl;
	particles = newParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
