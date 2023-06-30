/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include "root_finder.hpp"

#include <Eigen/Eigen>

#include <iostream>
#include <cmath>
#include <cfloat>
#include <vector>

typedef Eigen::Matrix<double, 3, 8> CoefficientMat;
typedef Eigen::Matrix<double, 3, 7> VelCoefficientMat;
typedef Eigen::Matrix<double, 3, 6> AccCoefficientMat;
typedef Eigen::Matrix<double, 2, 4> AngleCoefficientMat;

class Piece
{
private:
    double duration;
    CoefficientMat coeffMat;
    AngleCoefficientMat coeffMatAngle;
    // decreasing order coeffiecient

public:
    Piece() = default;

    Piece(double dur, const CoefficientMat &cMat, const AngleCoefficientMat &angleMat)
        : duration(dur), coeffMat(cMat), coeffMatAngle(angleMat) {}

    inline int getDim() const
    {
        return 5;  // x y z psi theta
    }

    inline int getOrder() const
    {
        return 7;
    }

    inline int getAngleOrder() const
    {
        return 3;
    }

    inline double getDuration() const
    {
        return duration;
    }

    inline const CoefficientMat &getCoeffMat() const
    {
        return coeffMat;
    }

    inline const AngleCoefficientMat &getAngleCoeffMat() const
    {
        return coeffMatAngle;
    }

    inline Eigen::Vector3d getPos(const double &t) const
    {
        Eigen::Vector3d pos(0.0, 0.0, 0.0);
        double tn = 1.0;
        for (int i = 7; i >= 0; i--)
        {
            pos += tn * coeffMat.col(i);
            tn *= t;
        }
        return pos;
    }

    inline Eigen::Vector3d getVel(const double &t) const
    {
        Eigen::Vector3d vel(0.0, 0.0, 0.0);
        double tn = 1.0;
        int n = 1;
        for (int i = 6; i >= 0; i--)
        {
            vel += n * tn * coeffMat.col(i);
            tn *= t;
            n++;
        }
        return vel;
    }

    inline Eigen::Vector3d getAcc(const double &t) const
    {
        Eigen::Vector3d acc(0.0, 0.0, 0.0);
        double tn = 1.0;
        int m = 1;
        int n = 2;
        for (int i = 5; i >= 0; i--)
        {
            acc += m * n * tn * coeffMat.col(i);
            tn *= t;
            m++;
            n++;
        }
        return acc;
    }

    inline Eigen::Vector3d getJer(const double &t) const
    {
        Eigen::Vector3d jer(0.0, 0.0, 0.0);
        double tn = 1.0;
        int l = 1;
        int m = 2;
        int n = 3;
        for (int i = 4; i >= 0; i--)
        {
            jer += l * m * n * tn * coeffMat.col(i);
            tn *= t;
            l++;
            m++;
            n++;
        }
        return jer;
    }

    inline Eigen::Vector2d getAngle(const double &t) const
    {
        Eigen::Vector2d angle(0.0, 0.0);
        double tn = 1.0;
        for(int i = 3; i >= 0; i--)
        {
            angle += tn * coeffMatAngle.col(i);
            tn *= t;
        }
        return angle;
    }

    inline Eigen::Vector2d getAngleRate(const double &t) const
    {
        Eigen::Vector2d rate(0.0, 0.0);
        double tn = 1.0;
        int n = 1;
        for(int i = 2; i >= 0; i--)
        {
            rate += n * tn * coeffMatAngle.col(i);
            tn *= t;
            n++;
        }
        return rate;
    }


};

class Trajectory
{
private:
    typedef std::vector<Piece> Pieces;
    Pieces pieces;

public:
    Trajectory() = default;

    Trajectory(const std::vector<double> &durs,
               const std::vector<CoefficientMat> &cMats,
               const std::vector<AngleCoefficientMat> &angleMats)
    {
        int N = std::min(durs.size(), cMats.size());
        pieces.reserve(N);
        for (int i = 0; i < N; i++)
        {
            pieces.emplace_back(durs[i], cMats[i], angleMats[i]);
        }
    }

    inline int getPieceNum() const
    {
        return pieces.size();
    }

    inline Eigen::VectorXd getDurations() const
    {
        int N = getPieceNum();
        Eigen::VectorXd durations(N);
        for (int i = 0; i < N; i++)
        {
            durations(i) = pieces[i].getDuration();
        }
        return durations;
    }

    inline double getTotalDuration() const
    {
        int N = getPieceNum();
        double totalDuration = 0.0;
        for (int i = 0; i < N; i++)
        {
            totalDuration += pieces[i].getDuration();
        }
        return totalDuration;
    }

    inline Eigen::Matrix3Xd getPositions() const
    {
        int N = getPieceNum();
        Eigen::Matrix3Xd positions(3, N + 1);
        for (int i = 0; i < N; i++)
        {
            positions.col(i) = pieces[i].getCoeffMat().col(5);
        }
        positions.col(N) = pieces[N - 1].getPos(pieces[N - 1].getDuration());
        return positions;
    }

    inline const Piece &operator[](int i) const
    {
        return pieces[i];
    }

    inline Piece &operator[](int i)
    {
        return pieces[i];
    }

    inline void clear(void)
    {
        pieces.clear();
        return;
    }

    inline Pieces::const_iterator begin() const
    {
        return pieces.begin();
    }

    inline Pieces::const_iterator end() const
    {
        return pieces.end();
    }

    inline Pieces::iterator begin()
    {
        return pieces.begin();
    }

    inline Pieces::iterator end()
    {
        return pieces.end();
    }

    inline void reserve(const int &n)
    {
        pieces.reserve(n);
        return;
    }

    inline void emplace_back(const Piece &piece)
    {
        pieces.emplace_back(piece);
        return;
    }

    inline void emplace_back(const double &dur,
                             const CoefficientMat &cMat,
                             const AngleCoefficientMat &yawMat)
    {
        pieces.emplace_back(dur, cMat, yawMat);
        return;
    }

    inline void append(const Trajectory &traj)
    {
        pieces.insert(pieces.end(), traj.begin(), traj.end());
        return;
    }

    inline int locatePieceIdx(double &t) const
    {
        int N = getPieceNum();
        int idx;
        double dur;
        for (idx = 0;
             idx < N &&
             t > (dur = pieces[idx].getDuration());
             idx++)
        {
            t -= dur;
        }
        if (idx == N)
        {
            idx--;
            t += pieces[idx].getDuration();
        }
        return idx;
    }

    inline Eigen::Vector3d getPos(double t) const
    {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getPos(t);
    }

    inline Eigen::Vector3d getVel(double t) const
    {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getVel(t);
    }

    inline Eigen::Vector3d getAcc(double t) const
    {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getAcc(t);
    }

    inline Eigen::Vector3d getJer(double t) const
    {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getJer(t);
    }

    inline Eigen::Vector2d getAngle(double t) const
    {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getAngle(t);
    }

    inline Eigen::Vector2d getAngleRate(double t) const
    {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getAngleRate(t);
    }

};

#endif