#ifndef __SegmentedTiming_H__
#define __SegmentedTiming_H__

#include <iostream>
#include <chrono>
#include <vector>
#include <string>

//easy way to make sure the timmgs print and computation are desactvated for final execution
#define ACTIVATE_SEGMENTED_TIMING

namespace SPH
{
	class SegmentedTiming
	{
	protected:
		std::string timer_name;
		std::vector<std::chrono::steady_clock::time_point> timepoints;
		std::vector<std::string> timepoints_names;
		std::vector<float> time;
		std::vector<float> cumul_time;
		int cur_point;
		int count_steps;
		bool saving;
		bool active;
		
	public:
	
#ifdef ACTIVATE_SEGMENTED_TIMING
		//contructor for the dynamic version
		SegmentedTiming(std::string timer_name_i, bool set_active=true);
		
		//constructor for static nbr of points
		SegmentedTiming(std::string timer_name_i,std::vector<std::string> names_i, bool set_active = true);


		//call that every start of timmings
		void init_step();
		
		//call that every start of timmings
		void time_next_point();
		
		//call that every end of timming
		void end_step();
	
		//this is if you want to do the timmings dynamically
		//if you want to be able to know the average values use the satically set version with the contructor
		void add_timestamp(std::string name);
		
		void recap_timings();
#else
		SegmentedTiming(std::string timer_name_i, bool set_active = true){}
		SegmentedTiming(std::string timer_name_i, std::vector<std::string> names_i, bool set_active = true){}

		void init_step() {}
		void time_next_point() {}
		void end_step() {}
		void add_timestamp(std::string name) {}
		void recap_timings() {}
#endif
	};
}

#endif