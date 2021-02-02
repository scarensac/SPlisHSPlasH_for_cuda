#ifndef __SegmentedTiming_H__
#define __SegmentedTiming_H__

#include <iostream>
#include <chrono>
#include <vector>
#include <string>

//easy way to make sure the timmgs print and computation are desactvated for final execution
#define ACTIVATE_SEGMENTED_TIMING


/*//an exeample of use
{
	std::vector<std::string> timing_names{ "p1","p2","p3" };
	static SPH::SegmentedTiming timings("timer name", timing_names, true);
	timings.init_step();//start point of the current step (if measuring avgs you need to call it at everystart of the loop)
	timings.time_next_point();//time p1
	timings.time_next_point();//time p2
	timings.time_next_point();//time p3
	timings.end_step();//end point of the current step (if measuring avgs you need to call it at every end of the loop)
	timings.recap_timings();//writte timming to cout
}


//*/
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

		//destructor
		~SegmentedTiming();

		//call that every start of timmings
		void init_step();
		
		//call that every start of timmings
		void time_next_point();
		
		//call that every end of timming
		void end_step(bool force_ending=false);
	
		//this is if you want to do the timmings dynamically
		//if you want to be able to know the average values use the satically set version with the contructor
		void add_timestamp(std::string name);
		
		//writte the results to the console
		void recap_timings();

		//get the avg for one particular timming
		float getTimmingAvg(int i);
#else
		SegmentedTiming(std::string timer_name_i, bool set_active = true){}
		SegmentedTiming(std::string timer_name_i, std::vector<std::string> names_i, bool set_active = true){}

		~SegmentedTiming() {}

		void init_step() {}
		void time_next_point() {}
		void end_step(bool force_ending = false) {}
		void add_timestamp(std::string name) {}
		void recap_timings() {}
		float getTimmingAvg(int i) {}
#endif
	};
}

#endif