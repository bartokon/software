#ifndef STRING_HPP
# define STRING_HPP
# include <iostream>
# include <cstring>
# include <cstddef>
# include <stdexcept>

namespace my {
  class string {
    private:
      char *mem;
      size_t len;

    public:
      // Default constructor
      string(char const *c_ptr = nullptr) {
        if (c_ptr == nullptr) {
          this->len = 0;
          this->mem = nullptr;
          return;
        }
        #ifdef DEBUG
          std::cout << "String constructor called" << std::endl;
        #endif
        for (len = 0; c_ptr[len] != '\0'; ++len); //TODO: Add error handling for length
        this->mem = new char[this->len + 1];
        std::memcpy(this->mem, c_ptr, this->len + 1);
      };

      // Copy constructor
      string(const string &other) {
        this->len = other.length();
        if (other.mem == nullptr) {
          this->mem = nullptr;
          return;
        }
        this->mem = new char[this->len + 1];
        std::memcpy(this->mem, other.c_str(), this->len + 1);
      };

      //Move constructor
      string(string &&other) noexcept {
        this->len = other.length();
        this->mem = other.mem;
        other.len = 0;
        other.mem = nullptr;
      };

      ~string() {
        #ifdef DEBUG
          std::cout << "String destructor called" << std::endl;
        #endif
        delete[] this->mem;
      };

      size_t length() const noexcept {
        return this->len;
      };

      char* c_str() const noexcept {
        return this->mem;
      };

      void print() const noexcept {
        if (this->mem == nullptr) {
          std::cout << "String: null" << std::endl;
        } else {
          std::cout << "String: " << this->mem << std::endl;
        }
      };

      // Comparison operator
      bool operator==(const string &other) const noexcept{
        if (this->length() != other.length()) return false;
        if (this->mem == nullptr && other.mem == nullptr) return true;
        if (this->mem == nullptr || other.mem == nullptr) return false;
        return std::memcmp(this->c_str(), other.c_str(), this->length()) == 0;
      };

      // Add operator
      string operator+(const string &other) const {
        if (this->length() == 0 || this->mem == nullptr) return string(other.c_str());
        if (other.length() == 0 || other.mem == nullptr) return string(this->c_str());
        size_t new_len = this->length() + other.length();
        char *new_mem = new char[new_len + 1];
        std::memcpy(new_mem, this->c_str(), this->length());
        std::memcpy(new_mem + this->length(), other.c_str(), other.length() + 1);
        string result(new_mem);
        delete[] new_mem;
        return result;
      };

      // Copy assignment operator
      string &operator=(const string &other) {
        if (this == &other) return *this;
        delete[] this->mem;
        this->len = other.length();
        if (other.mem == nullptr) {
          this->mem = nullptr;
        } else {
          this->mem = new char[this->len + 1];
          std::memcpy(this->mem, other.c_str(), this->len + 1);
        }
        return *this;
      };

      // Move assignment operator
      string &operator=(string &&other) noexcept {
        if (this == &other) return *this;
        delete[] this->mem;
        this->len = other.length();
        this->mem = other.mem;
        other.len = 0;
        other.mem = nullptr;
        return *this;
      };

      // Concatenation assignment operator
      string &operator+=(const string &other) {
        if (other.length() == 0) return *this;
        if (this->length() == 0) {
          *this = other;
          return *this;
        }
        size_t new_len = this->length() + other.length();
        char *new_mem = new char[new_len + 1];
        std::memcpy(new_mem, this->c_str(), this->length());
        std::memcpy(new_mem + this->length(), other.c_str(), other.length() + 1);
        delete[] this->mem;
        this->mem = new_mem;
        this->len = new_len;
        return *this;
      };

      // Ostream operator
      friend std::ostream& operator<<(std::ostream &os, const string &s) {
        return s.mem == nullptr ? os : os << s.c_str();
      };
  };

};
#endif