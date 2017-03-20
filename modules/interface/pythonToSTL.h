/*
 * pythonToSTL.h
 *
 *  Created on: 8 Mar 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_MODULES_INTERFACE_PYTHONTOSTL_H_
#define DEEPJET_MODULES_INTERFACE_PYTHONTOSTL_H_
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
#include <vector>
#include "TString.h"
#include <string>

template<class T>
std::vector<T> toSTLVector(const boost::python::list lin){
	std::vector<T>  out(boost::python::len(lin));
	for(size_t i=0;i<boost::python::len(lin);i++)
		out.at(i)=boost::python::extract<T>(lin[i]);
	return out;
}

template<>
std::vector<TString> toSTLVector(const boost::python::list lin){
	std::vector<TString>  out(boost::python::len(lin));
	for(size_t i=0;i<boost::python::len(lin);i++){
		std::string stdstr=boost::python::extract<std::string>(lin[i]);
		out.at(i)=stdstr.data();
	}
	return out;
}

template<class T>
std::vector<std::vector<T> > toSTL2DVector(const boost::python::list lin){
	std::vector<std::vector<T> > out;
	for(size_t i=0;i<boost::python::len(lin);i++){
		std::vector<T> tmp(boost::python::len(lin[i]));
		for(size_t j=0;j<boost::python::len(lin[i]);j++)
			tmp.at(j)=boost::python::extract<T>(lin[i][j]);
		out.push_back(tmp);
	}
	return out;
}


template<>
std::vector<std::vector<TString> > toSTL2DVector(const boost::python::list lin){
	std::vector<std::vector<TString> > out;
	for(size_t i=0;i<boost::python::len(lin);i++){
		std::vector<TString> tmp(boost::python::len(lin[i]));
		for(size_t j=0;j<boost::python::len(lin[i]);j++){
			std::string stdstr=boost::python::extract<std::string>(lin[i][j]);
			tmp.at(j)=stdstr.data();
		}
		out.push_back(tmp);
	}
	return out;
}


#endif /* DEEPJET_MODULES_INTERFACE_PYTHONTOSTL_H_ */
