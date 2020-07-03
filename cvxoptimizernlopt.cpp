/*
 * Solve the relaxed convex programming
 * problem using NLOpt API and toolbox
 */
#include <iomanip>
#include <iostream>
#include <vector>
#include <vector>
#include <exception>
#include <nlopt.hpp>
#ifndef TESTCVXBIN
#include <boost/python.hpp>
#endif
using namespace std;

/**
 * The optimization space is x = <c,p>
 * where c is the vector of allocations (integer variables)
 * adjoined with the power consumption (p) (continuous variables)
 */
class ptss_constraint_param {
    public :
        vector<double> a;
        vector<double> b;
        double deadline;
        unsigned int sel_idx; /* Index to be selected for separable coefficients */
        ptss_constraint_param(vector<double> a,vector<double> b,double d,unsigned int idx) : a(a), b(b), deadline(d), sel_idx(idx) {}
} ;

std::ostream& operator<<(std::ostream& os, const vector<double> &vi) {
    unsigned int i;
    os << "<";
    for (i = 0; i < vi.size(); i++) {
        os << vi[i] ;
        os << ",";
    }
    os << ">";
}

double ptss_func_pkp(const std::vector<double> &x, \
                 std::vector<double> &grad, \
                 void *my_func_data)
{
    if (!grad.empty()) {
        for (unsigned int i = 0; i < x.size()-1; i++) {
            grad[i] = 0.0;
        }
        grad[x.size()-1] = 1;
    }
    double f = x[x.size()-1];
    return f;
}


/* Dual Problem objective function */
double ptss_func_et(const std::vector<double> &x, \
                 std::vector<double> &grad, \
                 void *my_func_data) {

    ptss_constraint_param *p = reinterpret_cast<ptss_constraint_param*>(my_func_data);
    vector<double> a = p->a, b = p->b;
    double d2 = p->deadline;

    if (!grad.empty()) {
        for (unsigned int i = 0; i < x.size()-1; i++) {
            grad[i] = (-a[i])/(x[i]*(a[i]*log(x[i])+b[i])*(a[i]*log(x[i])+b[i]));
        }
        grad[x.size()-1] = 0.0;
    }
    /* Compute the constraint function */
    double f = 0.0;
    double tmp = 0.0;
    for (unsigned int i = 0; i < x.size()-1; i++) {
        tmp = 1/(a[i]*log(x[i])+b[i]);
        f += tmp;
    }
    return f;
}


/* Execution Time Constraints */
double ptss_constraint_exectime(const std::vector<double> &x, \
                                std::vector<double> &grad, \
                                void *param) {
    ptss_constraint_param *p = reinterpret_cast<ptss_constraint_param*>(param);
    vector<double> a = p->a, b = p->b;
    double d2 = p->deadline;
    if (!grad.empty()) {
        for (unsigned int i = 0; i < x.size()-1; i++) {
            grad[i] = (-a[i])/(a[i]*x[i]+b[i])*(a[i]*x[i]+b[i]);
        }
        grad[x.size()-1] = 0.0;
    }

    /* Compute the constraint function */
    double f = 0.0;
    double tmp = 0.0;
    for (unsigned int i = 0; i < x.size()-1; i++) {
        tmp = 1/(a[i]*x[i]+b[i]);
        f += tmp;
    }
    f = f-d2;
    static int count = 0;
    #ifdef TESTCVXBIN
    cout << "Execution Time Constraint_"<<count++<<"("<<x<<") = "<<f<<endl;
    #endif
    return f;
}

/* Minimax Power Objective converted into N additional constraints */
double ptss_constraint_power(const std::vector<double> &x, \
                             std::vector<double> &grad, \
                             void *param) {
    ptss_constraint_param *p = reinterpret_cast<ptss_constraint_param*>(param);
    vector<double> a = p->a, b = p->b;
    unsigned int idx = p->sel_idx;
    if (!grad.empty()) {
        for (unsigned int i = 0; i < x.size()-1; i++) {
            if (i == idx)
                grad[i] = a[i];
            else 
                grad[i] = 0.0;
        }
        grad[x.size()-1] = -1;
    }
    double f = a[idx]*x[idx] + b[idx] - x[x.size()-1];
    #ifdef TESTCVXBIN
    cout << "ptss_constraint_power:"<<idx;
    cout << x << std::endl;
    #endif
    return f;
}

double compute_pkpower(const vector<double> &x,const vector<double> &ap,const vector<double> &bp) {
    int i;
    double max = 0;
    double php = 0;
    for (i = 0; i < x.size()-1; i++) {
        php = ap[i]*x[i] + bp[i];
        if (php >= max) {
            max = php;
        }
    }
    return max;
}

double cpp_cvx_optimizer(
    int NPH,\
    vector<double> &x,\
    vector<double> &aet,\
    vector<double> &bet,\
    vector<double> &ap,\
    vector<double> &bp,\
    double D,\
    int llim,\
    int ulim) 
{
    // int NPH = x.size()-1;                // Number of phases
    nlopt::opt opt(nlopt::LD_MMA, NPH+1);

    /* Set the Box Constraints */
    std::vector<double> lb(NPH+1);
    std::vector<double> ub(NPH+1);
    for (int i = 0; i < NPH; i++) {
        lb[i] = llim;
        ub[i] = ulim;
    }
    lb[NPH] = 0.0;
    ub[NPH] = HUGE_VAL;
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    
    /* Set the Parameters for inequality constraints */
    ptss_constraint_param paramEt(aet,bet,D,-1);
    vector<ptss_constraint_param> param;
    for (int i = 0; i < NPH; i++) {
        param.push_back(ptss_constraint_param(ap,bp,D,i));
    }
    opt.add_inequality_constraint(ptss_constraint_exectime, &paramEt, 1e-8);
    for (int i = 0; i < NPH; i++) {
        opt.add_inequality_constraint(ptss_constraint_power, &param[i], 1e-8);
    }

    /* Set the objective function */
    opt.set_min_objective(ptss_func_pkp, NULL);

    /* Stopping Criteria and Initial Point*/
    opt.set_ftol_rel(1e-4);
    
    for (int i = 0; i < NPH; i++) {
        x[i] = ulim;
    }
    x[NPH] = 0.0;
    double tmp = compute_pkpower(x,ap,bp);
    x[NPH] = tmp;
    double minf = 0.0;
    
    try {
        int r = opt.optimize(x, minf);
        #ifdef TESTCVXBIN
        cout << x << std::endl;
        cout << r << std::endl;
        #endif
        return minf;
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }
    return minf;
}

#ifdef TESTCVXBIN
int main() {
    int NPH = 4;
    int LLIM = 1;
    int ULIM = 16;
    vector<double> x(NPH+1);
    vector<double> aet = {0.9306707270772263, 0.781253569721745, 1.96668139375962, 1.7920940392776321};
    vector<double> bet = {1.3267719165086922, 0.765264919613304, 1.3159522447618568, 0.5911815286630802};
    vector<double> ap  = {1.5736020733308043, 0.5113181889897235, 0.30285882145827975, 0.4394272763462338};
    vector<double> bp  = {0.43724930121492356, 1.4835425054235123, 1.1986166932894586, 0.862884848808483};
    double D = 2;

    cpp_cvx_optimizer(NPH,\
    x,\
    aet,\
    bet,\
    ap,\
    bp,\
    D,\
    LLIM,\
    ULIM);
}
#else

/* Wraps the previous function to ensure smooth export to python */
typedef vector<double> __phase_t;
namespace py=boost::python;

// Converts a C++ vector to a python list
template <class T>
py::list toPythonList(std::vector<T> vec) {
    typename std::vector<T>::iterator iter;
    py::list list;
    for (iter = vec.begin(); iter != vec.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}

template< typename T >
std::vector< T > toSTLVec( const py::object& iterable ) {
    return std::vector< T >( py::stl_input_iterator< T >( iterable ),
                             py::stl_input_iterator< T >( ) );
}

class CPPCVXOptimizer {
    private:
        int NPH;
        __phase_t x;
        __phase_t aet;
        __phase_t bet;
        __phase_t ap;
        __phase_t bp;
        int llim;
        int ulim;
    public:
        CPPCVXOptimizer() : NPH(-1) {}
        void setParams(
            int NPH,\
            py::list x,\
            py::list aet,\
            py::list bet,\
            py::list ap,\
            py::list bp,\
            int llim,\
            int ulim
        );
        double optimize(double D);
        py::list getOpt();
};

void CPPCVXOptimizer::setParams(
            int NPH,\
            py::list x,\
            py::list aet,\
            py::list bet,\
            py::list ap,\
            py::list bp,\
            int llim,\
            int ulim
        ) {
    this->NPH = NPH;
    this->llim = llim;
    this->ulim = ulim;

    this->x   = toSTLVec<double>(x);
    this->aet = toSTLVec<double>(aet);
    this->bet = toSTLVec<double>(bet);
    this->ap  = toSTLVec<double>(ap);
    this->bp  = toSTLVec<double>(bp);

}

py::list CPPCVXOptimizer::getOpt() {
    return toPythonList(this->x);
}

double CPPCVXOptimizer::optimize(double d) {
    if (this->NPH <= 1) {
        cout << "No phases found" << endl;
        return -HUGE_VAL;
    }
    double d2 = cpp_cvx_optimizer(
    this->NPH,\
    this->x,\
    this->aet,\
    this->bet,\
    this->ap,\
    this->bp,\
    d,\
    this->llim,\
    this->ulim);
    
    return d2;
}

BOOST_PYTHON_MODULE(testcvxopt) {
    py::class_<CPPCVXOptimizer>("CPPCVXOptimizer")
        .def("setParams", &CPPCVXOptimizer::setParams)
        .def("optimize", &CPPCVXOptimizer::optimize)
        .def("getOpt",&CPPCVXOptimizer::getOpt);
}
#endif