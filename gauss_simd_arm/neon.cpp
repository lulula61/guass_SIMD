#include <iostream>
#include <arm_neon.h>
#include <stdio.h>
#include <time.h>
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
void neon_unaligned_d()  //unaligned. Only the double loop is programmed by SIMD.
{
	int i,j,k;
        float32x4_t vt,va;
        for(k=0;k<N;k++)
        {
                vt=vdupq_n_f32(m[k][k]);
                for(j=k+1;j+4<=N;j+=4)
                {
                        va=vld1q_f32(m[k]+j);
                        va=vdivq_f32(va,vt);
                        vst1q_f32(m[k]+j,va);
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
void neon_unaligned_t()  //unaligned. Only the triple loop is SIMD programmed.
{
	int i,j,k;
	float32x4_t vaik,vakj,vaij,vx;
        for(k=0;k<N;k++)
        {
                for(j=k+1;j<N;j++)
                        m[k][j]=m[k][j]/m[k][k];
                m[k][k]=1.0;
		for(i=k+1;i<N;i++)
                {
                        vaik=vdupq_n_f32(m[i][k]);
                        for(j=k+1;j+4<=N;j+=4)
                        {
                                vakj=vld1q_f32(m[k]+j);
                                vaij=vld1q_f32(m[i]+j);
                                vx=vmulq_f32(vakj,vaik);
                                vaij=vsubq_f32(vaij,vx);
                                vst1q_f32(m[i]+j,vaij);
                        }
                        for(j=j-4;j<N;j++)
                        //Calculates several elements at the end of the line
                                m[i][j]=m[i][j]-m[i][k]*m[k][j];
                        m[i][k]=0;
                }
        }

}
void neon_unaligned()  //unaligned. The double cycle and triple cycle are programmed by SIMD.
{
	int i,j,k;
	float32x4_t vt,va,vaik,vakj,vaij,vx;
        for(k=0;k<N;k++)
        {
		vt=vdupq_n_f32(m[k][k]);
                for(j=k+1;j+4<=N;j+=4)
		{
			va=vld1q_f32(m[k]+j);
			va=vdivq_f32(va,vt);
			vst1q_f32(m[k]+j,va);
		}
		for(j=j-4;j<N;j++)
		//Calculates several elements at the end of the line
			m[k][j]=m[k][j]/m[k][k];
                m[k][k]=1.0;
                for(i=k+1;i<N;i++)
                {
			vaik=vdupq_n_f32(m[i][k]);
                        for(j=k+1;j+4<=N;j+=4)
			{
				vakj=vld1q_f32(m[k]+j);
				vaij=vld1q_f32(m[i]+j);
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);
				vst1q_f32(m[i]+j,vaij);
			}
			for(j=j-4;j<N;j++)
                	//Calculates several elements at the end of the line
				m[i][j]=m[i][j]-m[i][k]*m[k][j];
                        m[i][k]=0;
                }
        }

}
void neon_aligned_d() //aligned. Only the double loop is programmed by SIMD.
{
        int i,j,k;
        float32x4_t vt,va;
        for(k=0;k<N;k++)
        {
                vt=vdupq_n_f32(m[k][k]);
		for(j=k+1;(j%4!=0)&&j<N;j++)
                //Calculate the elements at the beginning of the line
                {
                        m[k][j]=m[k][j]/m[k][k];
                }
                for(;j+4<=N;j+=4)
                {
                        va=vld1q_f32(m[k]+j);
                        va=vdivq_f32(va,vt);
                        vst1q_f32(m[k]+j,va);
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
void neon_aligned_t()  //aligned. Only the triple loop is SIMD programmed.
{
        int i,j,k;
        float32x4_t vaik,vakj,vaij,vx;
        for(k=0;k<N;k++)
        {
                for(j=k+1;j<N;j++)
                        m[k][j]=m[k][j]/m[k][k];
                m[k][k]=1.0;
		for(i=k+1;i<N;i++)
                {
                        vaik=vdupq_n_f32(m[i][k]);
                        for(j=k+1;(j%4!=0)&&j<N;j++)
                        //Calculate the elements at the beginning of the line
                        {
                                m[i][j]=m[i][j]-m[i][k]*m[k][j];
                        }
                        for(;j+4<=N;j+=4)
                        {
                                vakj=vld1q_f32(m[k]+j);
                                vaij=vld1q_f32(m[i]+j);
                                vx=vmulq_f32(vakj,vaik);
                                vaij=vsubq_f32(vaij,vx);
                                vst1q_f32(m[i]+j,vaij);
                        }
                        for(j=j-4;j<N;j++)
                        //Calculates several elements at the end of the line
                                m[i][j]=m[i][j]-m[i][k]*m[k][j];
                        m[i][k]=0;
                }
        }

}

void neon_aligned()    //aligned. The double cycle and triple cycle are programmed by SIMD.
{
	int i,j,k;
        float32x4_t vt,va,vaik,vakj,vaij,vx;
        for(k=0;k<N;k++)
        {
                vt=vdupq_n_f32(m[k][k]);
		for(j=k+1;(j%4!=0)&&j<N;j++)
		//Calculate the elements at the beginning of the line
		{
			m[k][j]=m[k][j]/m[k][k];
		}
                for(;j+4<=N;j+=4)
                {
                        va=vld1q_f32(m[k]+j);
                        va=vdivq_f32(va,vt);
                        vst1q_f32(m[k]+j,va);
                }
                for(j=j-4;j<N;j++)
                //Calculates several elements at the end of the line
                        m[k][j]=m[k][j]/m[k][k];
                m[k][k]=1.0;
                for(i=k+1;i<N;i++)
                {
                        vaik=vdupq_n_f32(m[i][k]);
			for(j=k+1;(j%4!=0)&&j<N;j++)
			//Calculate the elements at the beginning of the line
			{
                        	m[i][j]=m[i][j]-m[i][k]*m[k][j];
                	}
                        for(;j+4<=N;j+=4)
			{
                                vakj=vld1q_f32(m[k]+j);
                                vaij=vld1q_f32(m[i]+j);
                                vx=vmulq_f32(vakj,vaik);
                                vaij=vsubq_f32(vaij,vx);
                                vst1q_f32(m[i]+j,vaij);
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
	struct timespec sts,ets;
	int step=5;
	int change=0;
	cout<<"sequential:"<<endl;
	for(N=5;N<=4000;N+=step)
	{
		if(N==10)
			step=10;
		if(N==100)
			step=100;
		if(N==1000)
			step=1000;
		m_reset();
		timespec_get(&sts,TIME_UTC);
		//to measure
		if(change==0)
			seq();
		else if(change==1)
			neon_unaligned_d();
		else if(change==2)
			neon_unaligned_t();
		else if(change==3)
			neon_unaligned();
		else if(change==4)
			neon_aligned_d();
		else if(change==5)
			neon_aligned_t();
		else
			neon_aligned();
		timespec_get(&ets,TIME_UTC);
		time_t dsec=ets.tv_sec-sts.tv_sec;
		long dnsec=ets.tv_nsec-sts.tv_nsec;
		if(dnsec<0)
		{
			dsec--;
			dnsec+=1000000000ll;
		}
		printf("%lld.%09llds\n",dsec,dnsec);
		if(N==4000)
		{
			change++;
			N=0;
			step=5;
			if(change==1)
        	                cout<<"neon_unaligned_d:"<<endl;
	                else if(change==2)
                        	cout<<"neon_unaligned_t:"<<endl;
                	else if(change==3)
                        	cout<<"neon_unaligned:"<<endl;
                	else if(change==4)
                        	cout<<"neon_aligned_d:"<<endl;
			else if(change==5)
                        	cout<<"neon_aligned_t:"<<endl;
                	else if(change==6)
                        	cout<<"neon_aligned:"<<endl;
			else
				break;
		}
	}
	return 0;
}
