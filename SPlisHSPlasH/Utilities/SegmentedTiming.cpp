#include "SegmentedTiming.h"

using namespace SPH;


#ifdef ACTIVATE_SEGMENTED_TIMING

SegmentedTiming::SegmentedTiming(std::string timer_name_i, bool set_active) {
	active = set_active;
	if (active) {
		timer_name = timer_name_i;
		count_steps = 0;
		saving = true;
		timepoints.push_back(std::chrono::steady_clock::now());
		cur_point = 1;
	}
}

SegmentedTiming::SegmentedTiming(std::string timer_name_i, std::vector<std::string> names_i, bool set_active) {
	active = set_active;
	if (active) {
		timer_name = timer_name_i;
		timepoints_names = names_i;
		timepoints.resize(timepoints_names.size() + 1);
		time.resize(timepoints_names.size() + 1, 0);
		cumul_time.resize(timepoints_names.size() + 1, 0);
		count_steps = 0;
		saving = false;
	}
}

void SegmentedTiming::init_step(){
	if (active) {
		cur_point=0;
		time_next_point();
		saving=true;
	}
}

void SegmentedTiming::time_next_point(){
	if (active) {
		timepoints[cur_point]=std::chrono::steady_clock::now();
		cur_point++;
	}
}

void SegmentedTiming::end_step(){
	if (active) {
		if(cur_point!=timepoints.size()){
			std::string msg("SegmentedTiming::end_step nbr of registered sampling do not fit the nbr of call");
			std::cout << msg << std::endl;
			throw(msg);
		}
		if(cur_point<2){
			std::string msg("SegmentedTiming::end_step no timing points have been registered");
			std::cout << msg << std::endl;
			throw(msg);
		}
	
		//read the times and update the avgs
		float t_total=0;
		for (int i=0; i<timepoints_names.size();++i){
			float t = std::chrono::duration_cast<std::chrono::nanoseconds> (timepoints[i+1] - timepoints[i]).count() / 1000000.0f;
			time[i]=t;
			cumul_time[i] += t;
			t_total+=t;
		}
		time.back()=t_total;
		cumul_time.back()+=t_total;
	

		count_steps++;
		saving=false;
	}
}

void SegmentedTiming::add_timestamp(std::string name){
	if (active) {
		timepoints.push_back(std::chrono::steady_clock::now());
		timepoints_names.push_back(name);
		cur_point++;
	}
}

void SegmentedTiming::recap_timings(){
	if (active) {
		if(saving){
			std::string msg("SegmentedTiming::recap_timmings() you must call end_step() before trying to print result once init_step() has been called");
			std::cout << msg << std::endl;
			throw(msg);
		}
	
		std::cout << " timer " << timer_name <<"  iter:  "<< count_steps<<"  total: " << (cumul_time.back() / count_steps) << "  (" << time.back() << ")" << std::endl;
		for (int i=0; i<timepoints_names.size();++i){
			std::cout << timepoints_names[i] << "  :" << (cumul_time[i] / count_steps) << "  (" << time[i] << ")" << std::endl;
		}
	}
}

#endif