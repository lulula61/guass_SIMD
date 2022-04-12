#include <iostream>
#include <nmmintrin.h>
#include <windows.h>
#include<stdlib.h>
using namespace std;
int N=0;
float** m;
void m_reset()
{
	m=new float*[N];
	for(int i=0;i<N;i++)
		m[i]=new float[N];
	for(int i=0;i<N;i++)
       	{
	       	for(int j=0;j<i;j++)
			m[i][ j]=0;
		m[i][ i]=1.0;
		for(int j=i+1;j<N;j++)
			m[i][ j]=rand();
	}
	for(int k=0;k<N;k++)
		for(int i=k+1;i<N;i++)
			for(int j=0;j<N;j++)
				m[i][ j]+=m[k][j];
}
void seq()
{
	int i,j,k;
	for(k=0;k<N;k++)
	{
		for(j=k+1;j<N;j++)
			m[k][j]=m[k][j]/m[k][k];
		m[k][k]=1.0;
		for(i=k+1;i<N;i++)
		{
			for(j=k+1;j<N;j++)
				m[i][j]=m[i][j]-m[i][k]*m[k][j];
			m[i][k]=0;
		}
	}
}
void sse_unaligned_d()  //unaligned. Only the double loop is programmed by SIMD.
{
	int i,j,k;
    __m128 vt,va;
    for(k=0;k<N;k++)
    {
        vt=_mm_set_ps(m[k][k],m[k][k],m[k][k],m[k][k]);
        for(j=k+1;j+4<=N;j+=4)
        {
            va=_mm_loadu_ps(m[k]+j);
            va=_mm_div_ps(va,vt);
            _mm_storeu_ps(m[k]+j,va);
        }
        for(j=j-4;j<N;j++)
        //Calculates several elements at the end of the line
            m[k][j]=m[k][j]/m[k][k];
        m[k][k]=1.0;
        for(i=k+1;i<N;i++)
        {
            for(j=k+1;j<N;j++)
                m[i][j]=m[i][j]-m[i][k]*m[k][j];
            m[i][k]=0;
        }
    }

}
void sse_unaligned_t()  //unaligned. Only the triple loop is SIMD programmed.
{
	int i,j,k;
	__m128 vaik,vakj,vaij,vx;
    for(k=0;k<N;k++)
    {
        for(j=k+1;j<N;j++)
            m[k][j]=m[k][j]/m[k][k];
        m[k][k]=1.0;
		for(i=k+1;i<N;i++)
        {
            vaik=_mm_set_ps(m[i][k],m[i][k],m[i][k],m[i][k]);
            for(j=k+1;j+4<=N;j+=4)
            {
                vakj=_mm_loadu_ps(m[k]+j);
                vaij=_mm_loadu_ps(m[i]+j);
                vx=_mm_mul_ps(vakj,vaik);
                vaij=_mm_sub_ps(vaij,vx);
                _mm_storeu_ps(m[i]+j,vaij);
            }
            for(j=j-4;j<N;j++)
            //Calculates several elements at the end of the line
                m[i][j]=m[i][j]-m[i][k]*m[k][j];
            m[i][k]=0;
        }
    }

}
void sse_unaligned()  //unaligned. The double cycle and triple cycle are programmed by SIMD.
{
	int i,j,k;
    __m128 vt,va,vaik,vakj,vaij,vx;
    for(k=0;k<N;k++)
    {
        vt=_mm_set_ps(m[k][k],m[k][k],m[k][k],m[k][k]);
        for(j=k+1;j+4<=N;j+=4)
        {
            va=_mm_loadu_ps(m[k]+j);
            va=_mm_div_ps(va,vt);
            _mm_storeu_ps(m[k]+j,va);
        }
        for(j=j-4;j<N;j++)
        //Calculates several elements at the end of the line
            m[k][j]=m[k][j]/m[k][k];
        m[k][k]=1.0;
        for(i=k+1;i<N;i++)
        {
            vaik=_mm_set_ps(m[i][k],m[i][k],m[i][k],m[i][k]);
            for(j=k+1;j+4<=N;j+=4)
            {
                vakj=_mm_loadu_ps(m[k]+j);
                vaij=_mm_loadu_ps(m[i]+j);
                vx=_mm_mul_ps(vakj,vaik);
                vaij=_mm_sub_ps(vaij,vx);
                _mm_storeu_ps(m[i]+j,vaij);
            }
            for(j=j-4;j<N;j++)
            //Calculates several elements at the end of the line
                m[i][j]=m[i][j]-m[i][k]*m[k][j];
            m[i][k]=0;
        }
    }

}
void sse_aligned_d() //aligned. Only the double loop is programmed by SIMD.
{
        int i,j,k;
        __m128 vt,va;
        for(k=0;k<N;k++)
        {
            vt=_mm_set_ps(m[k][k],m[k][k],m[k][k],m[k][k]);
            for(j=k+1;(j%4!=0)&&j<N;j++)
            //Calculate the elements at the beginning of the line
            {
                m[k][j]=m[k][j]/m[k][k];
            }
            for(;j+4<=N;j+=4)
            {
                va=_mm_load_ps(m[k]+j);
                va=_mm_div_ps(va,vt);
                _mm_store_ps(m[k]+j,va);
            }
            for(j=j-4;j<N;j++)
            //Calculates several elements at the end of the line
                m[k][j]=m[k][j]/m[k][k];
            m[k][k]=1.0;
            for(i=k+1;i<N;i++)
            {
                for(j=k+1;j<N;j++)
                    m[i][j]=m[i][j]-m[i][k]*m[k][j];
                m[i][k]=0;
            }
        }
}
void sse_aligned_t()  //aligned. Only the triple loop is SIMD programmed.
{
        int i,j,k;
        __m128 vaik,vakj,vaij,vx;
        for(k=0;k<N;k++)
        {
            for(j=k+1;j<N;j++)
                m[k][j]=m[k][j]/m[k][k];
            m[k][k]=1.0;
            for(i=k+1;i<N;i++)
            {
                vaik=_mm_set_ps(m[i][k],m[i][k],m[i][k],m[i][k]);
                for(j=k+1;(j%4!=0)&&j<N;j++)
                //Calculate the elements at the beginning of the line
                {
                    m[i][j]=m[i][j]-m[i][k]*m[k][j];
                }
                for(;j+4<=N;j+=4)
                {
                    vakj=_mm_load_ps(m[k]+j);
                    vaij=_mm_load_ps(m[i]+j);
                    vx=_mm_mul_ps(vakj,vaik);
                    vaij=_mm_sub_ps(vaij,vx);
                    _mm_store_ps(m[i]+j,vaij);
                }
                for(j=j-4;j<N;j++)
                //Calculates several elements at the end of the line
                    m[i][j]=m[i][j]-m[i][k]*m[k][j];
                m[i][k]=0;
            }
        }

}

void sse_aligned()    //aligned. The double cycle and triple cycle are programmed by SIMD.
{
        int i,j,k;
        __m128 vt,va,vaik,vakj,vaij,vx;
        for(k=0;k<N;k++)
        {
            vt=_mm_set_ps(m[k][k],m[k][k],m[k][k],m[k][k]);
            for(j=k+1;(j%4!=0)&&j<N;j++)
            //Calculate the elements at the beginning of the line
            {
                m[k][j]=m[k][j]/m[k][k];
            }
            for(;j+4<=N;j+=4)
            {
                va=_mm_load_ps(m[k]+j);
                va=_mm_div_ps(va,vt);
                _mm_store_ps(m[k]+j,va);
            }
            for(j=j-4;j<N;j++)  //Calculates several elements at the end of the line
                m[k][j]=m[k][j]/m[k][k];
            m[k][k]=1.0;
            for(i=k+1;i<N;i++)
            {
                vaik=_mm_set_ps(m[i][k],m[i][k],m[i][k],m[i][k]);
                for(j=k+1;(j%4!=0)&&j<N;j++)
                //Calculate the elements at the beginning of the line
                {
                    m[i][j]=m[i][j]-m[i][k]*m[k][j];
                }
                for(;j+4<=N;j+=4)
                {
                    vakj=_mm_load_ps(m[k]+j);
                    vaij=_mm_load_ps(m[i]+j);
                    vx=_mm_mul_ps(vakj,vaik);
                    vaij=_mm_sub_ps(vaij,vx);
                    _mm_store_ps(m[i]+j,vaij);
                }
                for(j=j-4;j<N;j++)
                //Calculates several elements at the end of the line
                    m[i][j]=m[i][j]-m[i][k]*m[k][j];
                m[i][k]=0;
            }
        }

}
int main()
{
	long long head, tail , freq ;
	int change=0,i;
	int n[10]={10,50,100,200,300,500,1000,2000,3000,4000};
	cout<<"sequential:"<<endl;
	for(i=0;i<2;i++)
	{
		N=n[i];
		m_reset();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq ) ;
        // start time
        QueryPerformanceCounter((LARGE_INTEGER* )&head) ;
		//to measure
		if(change==0)
			seq();
		else if(change==1)
			sse_unaligned_d();
		else if(change==2)
			sse_unaligned_t();
		else if(change==3)
			sse_unaligned();
		else if(change==4)
			sse_aligned_d();
		else if(change==5)
			sse_aligned_t();
		else
			sse_aligned();
		// end time
        QueryPerformanceCounter((LARGE_INTEGER *)&tail ) ;
        cout << (tail-head) * 1000.0 / freq << endl;
		if(i==9)
		{
			change++;
			i=-1;
			if(change==1)
                cout<<"sse_unaligned_d:"<<endl;
            else if(change==2)
                cout<<"sse_unaligned_t:"<<endl;
            else if(change==3)
                cout<<"sse_unaligned:"<<endl;
            else if(change==4)
                cout<<"sse_aligned_d:"<<endl;
			else if(change==5)
                cout<<"sse_aligned_t:"<<endl;
            else if(change==6)
                cout<<"sse_aligned:"<<endl;
			else
				break;
		}
		for(int j=0;j<N;j++)
            delete[] m[j];
        delete[]m;
	}
	return 0;
}
