#include <iostream>
#include <random>

#include <string.hpp>
#include <string>

void print_compare(const my::string &s1, const std::string &s2){
    //Compare lengths
    std::cout << "my::string Length: " << s1.length() << std::endl;
    std::cout << "std::string length: " << s2.length() << std::endl;
    //Print by calling the overloaded operator<<
    std::cout << "my::string: " << s1 << std::endl;
    std::cout << "std::string: " << s2 << std::endl;
}

int main (void) {
    constexpr char sample_text[] = "Hello World!";
    std::cout << "/*************/" << std::endl;
    my::string s1(sample_text);
    std::string s2(sample_text);
    print_compare(s1, s2);
    std::cout << "/*************/" << std::endl;
    my::string s3 = s1 + s1;
    std::string s4 = s2 + s2;
    print_compare(s3, s4);
    std::cout << "/*************/" << std::endl;
    s3 += s3;
    s4 += s4;
    print_compare(s3, s4);
    std::cout << "/*************/" << std::endl;
    s3 = s3 + "#";
    s4 = s4 + "#";
    print_compare(s3, s4);
    return 0;
}
