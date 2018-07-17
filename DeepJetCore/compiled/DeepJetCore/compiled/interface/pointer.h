/*
 * pointer.h
 *
 *  Created on: 8 Mar 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_MODULES_INTERFACE_POINTER_H_
#define DEEPJET_MODULES_INTERFACE_POINTER_H_
#include <boost/python.hpp>

namespace py = boost::python;


template<typename POINTER_TYPE>
class Pointer
{
public:
    Pointer(POINTER_TYPE p = nullptr): value(p)
    {
    }

    operator POINTER_TYPE() const
    {
        return value;
    }

    template<typename T>
    T get() const
    {
        return reinterpret_cast<T>(value);
    }

    class Converter
    {
    public:
        Converter()
        {
            py::to_python_converter<Pointer, Converter>();
            py::converter::registry::push_back(&Converter::convertable, &Converter::construct, py::type_id<Pointer>());
            if (ctypes_c_void_p.is_none())
            {
                ctypes_c_void_p = py::import("ctypes").attr("c_void_p");
            }
        }

        static void* convertable(PyObject *obj_ptr)
        {
            return PyObject_IsInstance(obj_ptr, ctypes_c_void_p.ptr())? obj_ptr: nullptr;
        }

        static void construct(PyObject *obj_ptr, py::converter::rvalue_from_python_stage1_data *data)
        {
            // From ctypes.c_void_p
            auto storage = reinterpret_cast<py::converter::rvalue_from_python_storage<Pointer>*>(data)->storage.bytes;
            py::object value_obj = py::object(py::handle<>(py::borrowed(obj_ptr))).attr("value");
            new(storage) Pointer(value_obj.is_none()? 0: reinterpret_cast<POINTER_TYPE>(uintptr_t(py::extract<uintptr_t>(value_obj))));
            data->convertible = storage;
        }

        static PyObject* convert(Pointer const &ptr)
        {
            return py::incref(ctypes_c_void_p(ptr.get<uintptr_t>()).ptr());
        }

    private:
        static py::object ctypes_c_void_p;
    };

    POINTER_TYPE value;
};

template<typename POINTER_TYPE>
py::object Pointer<POINTER_TYPE>::Converter::ctypes_c_void_p;

typedef float* PMYSTRUCT;
typedef Pointer<PMYSTRUCT> FloatPtr;
// from http://stackoverflow.com/questions/30591085/boost-python-converter-for-pointers-doesnt-work

#endif /* DEEPJET_MODULES_INTERFACE_POINTER_H_ */
