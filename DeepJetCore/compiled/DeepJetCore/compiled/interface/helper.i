%module helper
%{
extern TString prependXRootD(const TString& path);
extern bool isApprox(const float& a , const float& b, float eps=0.001);
extern float deltaPhi(const float& phi1, const float& phi2);
%}
extern TString prependXRootD(const TString& path);
extern bool isApprox(const float& a , const float& b, float eps=0.001);
extern float deltaPhi(const float& phi1, const float& phi2);
