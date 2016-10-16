/*
The function feature_hash implements the feature hashing trick as discussed in the paper


 References
        ----------
        [1] Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg (2009). Feature Hashing
        for Large Scale Multitask Learning. Proc. ICML.

Caveat: Use the code at your own risk
Contact: Chandresh Maurya for any bug or improvement
Email: ckm.jnu@gmail.com
*/


#include<iostream>
#include<armadillo>
#include<functional>//for hash function
#include "hash.h"

/***************************Namespaces***********************************************/
using namespace std;
using namespace arma;



void feature_hash(mat& train_cat,mat& cat_data, int output_dim)
{

  train_cat.zeros(cat_data.n_rows,output_dim);
   hash<double> myhash;
  for(uword i = 0;i< cat_data.n_rows;i++)
  {
       for(uword j=0;j<cat_data.n_cols;j++)
       {

        if(cat_data(i,j) != 0)
         {
         // cout<<" i,j="<<i<<","<<j<<endl;
           uint64_t value =  uniform_hash(&cat_data(i,j), sizeof(double),1);
           if(myhash(cat_data(i,j))%2 == 0)
            train_cat(i,value%output_dim) +=1;
           else
            train_cat(i,value%output_dim) -=1;

        }

       }

  }

}


int main()
{

  mat train_cat(4,5,fill::ones); //Assumption:: train_cat stores categorical string data, you should do ordinal encoding to convert strings to uniques numbers before trying feature_hash
  mat cat_data; //output hashed data
  int output_dim = cat_data.n_cols; // output feature dimensions you want to have or project
 feature_hash(train_cat,cat_data, output_dim);

}

