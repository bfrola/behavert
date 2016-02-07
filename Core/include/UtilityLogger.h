// ----------------------------------------------------------------------------
// This source file is part of BehaveRT 
// http://isis.dia.unisa.it/projects/behavert/
//
// Copyright (c) 2008-2010 ISISLab - University of Salerno
// Original author: Bernardino Frola <frola@dia.unisa.it>
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// ----------------
// Change log
//
// 12-08 bf: Created
//
// ----------------

#pragma once

#include <list>
#include "stdarg.h"

using namespace std;

namespace BehaveRT
{

	/**
		\author Bernardino Frola

		\brief This class provide a simple managment of a finite set of string, for logging purposes.

		
	 */
	class UtilityLogger
	{
	public:
		UtilityLogger( unsigned int maxSize );
		~UtilityLogger(void);

		enum mLogLevels {
			LOG_DEBUG, 
			LOG_INFO
		};
		
		typedef string msgSenderType;
		typedef string msgTextType;	

		typedef pair<msgSenderType, msgTextType> logMsg;
		typedef pair<logMsg, mLogLevels> logItem;
		
		typedef list<logItem> msgListType;
		typedef list<logItem>::iterator msgListIteratorType;
		
		/**
			This method log a given message.
			\param <msgSender> Message sender.
			\param <msgText> The message to log.
			\param <level> The log level. One of UtilityLogger::mLogLevels.
		*/
		void log( msgSenderType msgSender, msgTextType msgText, mLogLevels level = LOG_INFO );
		void quickLog( msgSenderType msgSender, const char *fmt, ...);
		
		/**
			\return True if the message senter is equal to the senter of the item.
		*/
		bool checkSender(logItem item);

		/**
			\return True if the message senter is greater or equal to the senter of the item.
		*/
		bool checkLevel(logItem item);

		/**
			\return The message enveloped into the item.
		*/
		string getMsgString(logItem item);
		
		/**
			Set the lof level.
		*/
		void setLevelString( string level );

		string getMsgListString( string separator );

		// ************************************************************************
		// Metodi GET/SET
		// ************************************************************************
		
		msgListType getMsgList() { return mMsgList; }
		msgListIteratorType getIterator() { return mMsgList_it; }
		
		bool isModified() { return mModified; }
		void setModified( bool modified ) { mModified = modified; }

		void setSenderFilter( string senderFilter ) { mSenderFilter = senderFilter; }
		string getSenderFilter( void ) { return mSenderFilter; }

		// ************************************************************************
		// Campi
		// ************************************************************************
	public:
		msgListType mMsgList;
		msgListIteratorType mMsgList_it;

	protected:
		unsigned int mMaxSize;
		
		mLogLevels mLevel;

		// Flag utilizzato per sapere quando deve essere aggiornato visualizzazione
		bool mModified;
		
		// Filtro sul campo sender dei messaggi di logging
		string mSenderFilter;


	};
};
 